from typing import Union

import torch

from ..image_object import Image
from ..window import Window
from .. import func
from ...utils.interpolate import interp2d
from ...param import forward


class SIPMixin:

    expect_ctype = (("RA---TAN-SIP",), ("DEC--TAN-SIP",))

    def __init__(
        self,
        *args,
        sipA={},
        sipB={},
        sipAP={},
        sipBP={},
        pixel_area_map=None,
        distortion_ij=None,
        distortion_IJ=None,
        filename=None,
        **kwargs,
    ):
        super().__init__(*args, filename=filename, **kwargs)
        if filename is not None:
            return
        self.sipA = sipA
        self.sipB = sipB
        self.sipAP = sipAP
        self.sipBP = sipBP

        self.update_distortion_model(
            distortion_ij=distortion_ij, distortion_IJ=distortion_IJ, pixel_area_map=pixel_area_map
        )

    @forward
    def pixel_to_plane(self, i, j, crtan, CD):
        di = interp2d(self.distortion_ij[0], j, i)
        dj = interp2d(self.distortion_ij[1], j, i)
        return func.pixel_to_plane_linear(i + di, j + dj, *self.crpix, CD, *crtan)

    @forward
    def plane_to_pixel(self, x, y, crtan, CD):
        I, J = func.plane_to_pixel_linear(x, y, *self.crpix, CD, *crtan)
        dI = interp2d(self.distortion_IJ[0], J, I)
        dJ = interp2d(self.distortion_IJ[1], J, I)
        return I + dI, J + dJ

    @property
    def pixel_area_map(self):
        return self._pixel_area_map

    def update_distortion_model(self, distortion_ij=None, distortion_IJ=None, pixel_area_map=None):
        """
        Update the pixel area map based on the current SIP coefficients.
        """

        # Pixelized distortion model
        #############################################################
        if distortion_ij is None or distortion_IJ is None:
            i, j = self.pixel_center_meshgrid()
            u, v = i - self.crpix[0], j - self.crpix[1]
            if distortion_ij is None:
                distortion_ij = torch.stack(func.sip_delta(u, v, self.sipA, self.sipB), dim=0)
            if distortion_IJ is None:
                # fixme maybe
                distortion_IJ = torch.stack(func.sip_delta(u, v, self.sipAP, self.sipBP), dim=0)
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
        self._pixel_area_map = A.abs()

    def copy(self, **kwargs):
        kwargs = {
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "pixel_area_map": self.pixel_area_map,
            **kwargs,
        }
        return super().copy(**kwargs)

    def blank_copy(self, **kwargs):
        kwargs = {
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "pixel_area_map": self.pixel_area_map,
            **kwargs,
        }
        return super().blank_copy(**kwargs)

    def get_window(self, other: Union[Image, Window], indices=None, **kwargs):
        """Get a sub-region of the image as defined by an other image on the sky."""
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        return super().get_window(
            other,
            pixel_area_map=self.pixel_area_map[indices],
            indices=indices,
            **kwargs,
        )

    def fits_info(self):
        info = super().fits_info()
        info["CTYPE1"] = "RA---TAN-SIP"
        info["CTYPE2"] = "DEC--TAN-SIP"
        for a, b in self.sipA:
            info[f"A_{a}_{b}"] = self.sipA[(a, b)]
        for a, b in self.sipB:
            info[f"B_{a}_{b}"] = self.sipB[(a, b)]
        for a, b in self.sipAP:
            info[f"AP_{a}_{b}"] = self.sipAP[(a, b)]
        for a, b in self.sipBP:
            info[f"BP_{a}_{b}"] = self.sipBP[(a, b)]
        return info

    def load(self, filename: str, hduext=0):
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

    def reduce(self, scale, **kwargs):
        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        return super().reduce(
            scale=scale,
            pixel_area_map=(
                self.pixel_area_map[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .sum(axis=(1, 3))
            ),
            distortion_ij=(
                self.distortion_ij[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .mean(axis=(1, 3))
            ),
            distortion_IJ=(
                self.distortion_IJ[: MS * scale, : NS * scale]
                .reshape(MS, scale, NS, scale)
                .mean(axis=(1, 3))
            ),
            **kwargs,
        )
