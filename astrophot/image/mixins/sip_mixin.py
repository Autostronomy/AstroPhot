from typing import Union

from ..image_object import Image
from ..window import Window
from .. import func
from ...utils.interpolate import interp2d
from ...param import forward


class SIPMixin:

    def __init__(self, *args, sipA=(), sipB=(), sipAP=(), sipBP=(), pixel_area_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sipA = sipA
        self.sipB = sipB
        self.sipAP = sipAP
        self.sipBP = sipBP

        i, j = self.pixel_center_meshgrid()
        u, v = i - self.crpix[0], j - self.crpix[1]
        self.distortion_ij = func.sip_delta(u, v, self.sipA, self.sipB)
        self.distortion_IJ = func.sip_delta(u, v, self.sipAP, self.sipBP)  # fixme maybe

        if pixel_area_map is None:
            self.update_pixel_area_map()
        else:
            self._pixel_area_map = pixel_area_map

    @forward
    def pixel_to_plane(self, i, j, crtan, pixelscale):
        di = interp2d(self.distortion_ij[0], i, j)
        dj = interp2d(self.distortion_ij[1], i, j)
        return func.pixel_to_plane_linear(i + di, j + dj, *self.crpix, pixelscale, *crtan)

    @forward
    def plane_to_pixel(self, x, y, crtan):
        I, J = func.plane_to_pixel_linear(x, y, *self.crpix, self.pixelscale_inv, *crtan)
        dI = interp2d(self.distortion_IJ[0], I, J)
        dJ = interp2d(self.distortion_IJ[1], I, J)
        return I + dI, J + dJ

    @property
    def pixel_area_map(self):
        return self._pixel_area_map

    def update_pixel_area_map(self):
        """
        Update the pixel area map based on the current SIP coefficients.
        """
        i, j = self.pixel_corner_meshgrid()
        x, y = self.pixel_to_plane(i, j)

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
            info[f"A{a}_{b}"] = self.sipA[(a, b)]
        for a, b in self.sipB:
            info[f"B{a}_{b}"] = self.sipB[(a, b)]
        for a, b in self.sipAP:
            info[f"AP{a}_{b}"] = self.sipAP[(a, b)]
        for a, b in self.sipBP:
            info[f"BP{a}_{b}"] = self.sipBP[(a, b)]
        return info

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
