from typing import Union, Optional, Tuple

from ..image_object import Image
from ..window import Window
from .. import func
from ... import config
from ...backend_obj import backend, ArrayLike
from ...utils.interpolate import interp2d
from ...param import forward


class SIPMixin:
    """A mixin class for SIP (Simple Image Polynomial) distortion model."""

    expect_ctype = (("RA---TAN-SIP",), ("DEC--TAN-SIP",))

    def __init__(
        self,
        *args,
        sipA: dict[Tuple[int, int], float] = {},
        sipB: dict[Tuple[int, int], float] = {},
        sipAP: dict[Tuple[int, int], float] = {},
        sipBP: dict[Tuple[int, int], float] = {},
        pixel_area_map: Optional[ArrayLike] = None,
        distortion_ij: Optional[ArrayLike] = None,
        distortion_IJ: Optional[ArrayLike] = None,
        filename: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, filename=filename, **kwargs)
        if filename is not None:
            return
        self.sipA = sipA
        self.sipB = sipB
        self.sipAP = sipAP
        self.sipBP = sipBP

        if len(self.sipAP) == 0 and len(self.sipA) > 0:
            self.compute_backward_sip_coefs()

        self.update_distortion_model(
            distortion_ij=distortion_ij, distortion_IJ=distortion_IJ, pixel_area_map=pixel_area_map
        )

    @forward
    def pixel_to_plane(
        self,
        i: ArrayLike,
        j: ArrayLike,
        crtan: ArrayLike,
        CD: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        di = interp2d(self.distortion_ij[0], i, j, padding_mode="border")
        dj = interp2d(self.distortion_ij[1], i, j, padding_mode="border")
        return func.pixel_to_plane_linear(i + di, j + dj, *self.crpix, CD, *crtan)

    @forward
    def plane_to_pixel(
        self,
        x: ArrayLike,
        y: ArrayLike,
        crtan: ArrayLike,
        CD: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        I, J = func.plane_to_pixel_linear(x, y, *self.crpix, CD, *crtan)
        dI = interp2d(self.distortion_IJ[0], I, J, padding_mode="border")
        dJ = interp2d(self.distortion_IJ[1], I, J, padding_mode="border")
        return I + dI, J + dJ

    @property
    def pixel_area_map(self):
        return self._pixel_area_map

    @property
    def A_ORDER(self) -> int:
        if self.sipA:
            return max(a + b for a, b in self.sipA)
        return 0

    @property
    def B_ORDER(self) -> int:
        if self.sipB:
            return max(a + b for a, b in self.sipB)
        return 0

    def compute_backward_sip_coefs(self):
        """
        Credit: Shu Liu and Lei Hi, see here:
        https://github.com/Roman-Supernova-PIT/sfft/blob/master/sfft/utils/CupyWCSTransform.py

        Compute the backward transformation from (U, V) to (u, v)
        """
        i, j = self.pixel_center_meshgrid()
        u, v = i - self.crpix[0], j - self.crpix[1]
        du, dv = func.sip_delta(u, v, self.sipA, self.sipB)
        U = (u + du).flatten()
        V = (v + dv).flatten()
        AP, BP = func.sip_backward_transform(
            u.flatten(), v.flatten(), U, V, self.A_ORDER, self.B_ORDER
        )
        self.sipAP = dict(
            ((p, q), ap.item()) for (p, q), ap in zip(func.sip_coefs(self.A_ORDER), AP)
        )
        self.sipBP = dict(
            ((p, q), bp.item()) for (p, q), bp in zip(func.sip_coefs(self.B_ORDER), BP)
        )

    def update_distortion_model(
        self,
        distortion_ij: Optional[ArrayLike] = None,
        distortion_IJ: Optional[ArrayLike] = None,
        pixel_area_map: Optional[ArrayLike] = None,
    ):
        """
        Update the pixel area map based on the current SIP coefficients.
        """

        # Pixelized distortion model
        #############################################################
        if distortion_ij is None or distortion_IJ is None:
            i, j = self.pixel_center_meshgrid()
            u, v = i - self.crpix[0], j - self.crpix[1]
            if distortion_ij is None:
                distortion_ij = backend.stack(func.sip_delta(u, v, self.sipA, self.sipB), dim=0)
            if distortion_IJ is None:
                # fixme maybe
                distortion_IJ = backend.stack(func.sip_delta(u, v, self.sipAP, self.sipBP), dim=0)
        self.distortion_ij = distortion_ij
        self.distortion_IJ = distortion_IJ

        # Pixel area map
        #############################################################
        if pixel_area_map is not None:
            self._pixel_area_map = pixel_area_map
            return
        i, j = self.pixel_corner_meshgrid()
        x, y = self.pixel_to_plane(i, j)

        # Shoelace formula for pixel area
        # 1: [:-1, :-1]
        # 2: [:-1, 1:]
        # 3: [1:, 1:]
        # 4: [1:, :-1]
        A = 0.5 * (
            x[:-1, :-1] * y[:-1, 1:]
            + x[:-1, 1:] * y[1:, 1:]
            + x[1:, 1:] * y[1:, :-1]
            + x[1:, :-1] * y[:-1, :-1]
            - (
                x[:-1, 1:] * y[:-1, :-1]
                + x[1:, 1:] * y[:-1, 1:]
                + x[1:, :-1] * y[1:, 1:]
                + x[:-1, :-1] * y[1:, :-1]
            )
        )
        self._pixel_area_map = backend.abs(A)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = config.DTYPE
        if device is None:
            device = config.DEVICE
        super().to(dtype=dtype, device=device)
        self._pixel_area_map = backend.to(self._pixel_area_map, dtype=dtype, device=device)
        self.distortion_ij = backend.to(self.distortion_ij, dtype=dtype, device=device)
        self.distortion_IJ = backend.to(self.distortion_IJ, dtype=dtype, device=device)

    def copy_kwargs(self, **kwargs):
        kwargs = {
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "pixel_area_map": self.pixel_area_map,
            "distortion_ij": self.distortion_ij,
            "distortion_IJ": self.distortion_IJ,
            **kwargs,
        }
        return super().copy_kwargs(**kwargs)

    def get_window(self, other: Union[Image, Window], indices=None, **kwargs):
        """Get a sub-region of the image as defined by an other image on the sky."""
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        return super().get_window(
            other,
            pixel_area_map=self.pixel_area_map[indices],
            distortion_ij=self.distortion_ij[:, indices[0], indices[1]],
            distortion_IJ=self.distortion_IJ[:, indices[0], indices[1]],
            indices=indices,
            **kwargs,
        )

    def fits_info(self):
        info = super().fits_info()
        info["CTYPE1"] = "RA---TAN-SIP"
        info["CTYPE2"] = "DEC--TAN-SIP"
        a_order = 0
        for a, b in self.sipA:
            info[f"A_{a}_{b}"] = self.sipA[(a, b)]
            a_order = max(a_order, a + b)
        info["A_ORDER"] = a_order
        b_order = 0
        for a, b in self.sipB:
            info[f"B_{a}_{b}"] = self.sipB[(a, b)]
            b_order = max(b_order, a + b)
        info["B_ORDER"] = b_order
        ap_order = 0
        for a, b in self.sipAP:
            info[f"AP_{a}_{b}"] = self.sipAP[(a, b)]
            ap_order = max(ap_order, a + b)
        info["AP_ORDER"] = ap_order
        bp_order = 0
        for a, b in self.sipBP:
            info[f"BP_{a}_{b}"] = self.sipBP[(a, b)]
            bp_order = max(bp_order, a + b)
        info["BP_ORDER"] = bp_order
        return info

    def load(self, filename: str, hduext: int = 0):
        hdulist = super().load(filename, hduext=hduext)
        self.sipA = {}
        if "A_ORDER" in hdulist[hduext].header:
            a_order = hdulist[hduext].header["A_ORDER"]
            for i in range(a_order + 1):
                for j in range(a_order + 1 - i):
                    key = (i, j)
                    if f"A_{i}_{j}" in hdulist[hduext].header:
                        self.sipA[key] = hdulist[hduext].header[f"A_{i}_{j}"]
        self.sipB = {}
        if "B_ORDER" in hdulist[hduext].header:
            b_order = hdulist[hduext].header["B_ORDER"]
            for i in range(b_order + 1):
                for j in range(b_order + 1 - i):
                    key = (i, j)
                    if f"B_{i}_{j}" in hdulist[hduext].header:
                        self.sipB[key] = hdulist[hduext].header[f"B_{i}_{j}"]
        self.sipAP = {}
        if "AP_ORDER" in hdulist[hduext].header:
            ap_order = hdulist[hduext].header["AP_ORDER"]
            for i in range(ap_order + 1):
                for j in range(ap_order + 1 - i):
                    key = (i, j)
                    if f"AP_{i}_{j}" in hdulist[hduext].header:
                        self.sipAP[key] = hdulist[hduext].header[f"AP_{i}_{j}"]
        self.sipBP = {}
        if "BP_ORDER" in hdulist[hduext].header:
            bp_order = hdulist[hduext].header["BP_ORDER"]
            for i in range(bp_order + 1):
                for j in range(bp_order + 1 - i):
                    key = (i, j)
                    if f"BP_{i}_{j}" in hdulist[hduext].header:
                        self.sipBP[key] = hdulist[hduext].header[f"BP_{i}_{j}"]
        self.update_distortion_model()
        return hdulist
