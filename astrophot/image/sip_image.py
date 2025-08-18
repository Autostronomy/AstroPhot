from typing import Tuple, Union

from .target_image import TargetImage
from .model_image import ModelImage
from .mixins import SIPMixin
from ..backend_obj import backend, ArrayLike
from .. import config


class SIPModelImage(SIPMixin, ModelImage):
    """
    A ModelImage with SIP distortion coefficients."""

    def crop(self, pixels: Union[int, Tuple[int, int], Tuple[int, int, int, int]], **kwargs):
        """
        Crop the image by the number of pixels given. This will crop
        the image in all four directions by the number of pixels given.
        """
        if isinstance(pixels, int):  # same crop in all dimension
            crop = (slice(pixels, -pixels), slice(pixels, -pixels))
        elif len(pixels) == 1:  # same crop in all dimension
            crop = (slice(pixels[0], -pixels[0]), slice(pixels[0], -pixels[0]))
        elif len(pixels) == 2:  # different crop in each dimension
            crop = (
                slice(pixels[1], -pixels[1]),
                slice(pixels[0], -pixels[0]),
            )
        elif len(pixels) == 4:  # different crop on all sides
            crop = (
                slice(pixels[0], -pixels[1]),
                slice(pixels[2], -pixels[3]),
            )
        else:
            raise ValueError(
                f"Invalid crop shape {pixels}, must be int, (int,), (int, int), or (int, int, int, int)!"
            )
        kwargs = {
            "pixel_area_map": self.pixel_area_map[crop],
            "distortion_ij": self.distortion_ij[:, crop[0], crop[1]],
            "distortion_IJ": self.distortion_IJ[:, crop[0], crop[1]],
            **kwargs,
        }
        return super().crop(pixels, **kwargs)

    def reduce(self, scale: int, **kwargs):
        """This operation will downsample an image by the factor given. If
        scale = 2 then 2x2 blocks of pixels will be summed together to
        form individual larger pixels. A new image object will be
        returned with the appropriate pixelscale and data tensor. Note
        that the window does not change in this operation since the
        pixels are condensed, but the pixel size is increased
        correspondingly.

        **Args:**
        -  `scale`: factor by which to condense the image pixels. Each scale X scale region will be summed [int]

        """
        if not isinstance(scale, int) and not (
            isinstance(scale, ArrayLike) and scale.dtype is backend.int32
        ):
            raise SpecificationConflict(f"Reduce scale must be an integer! not {type(scale)}")
        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        kwargs = {
            "pixel_area_map": (
                backend.sum(
                    self.pixel_area_map[: MS * scale, : NS * scale].reshape(MS, scale, NS, scale),
                    dim=(1, 3),
                )
            ),
            "distortion_ij": (
                backend.mean(
                    self.distortion_ij[:, : MS * scale, : NS * scale].reshape(
                        2, MS, scale, NS, scale
                    ),
                    dim=(2, 4),
                )
            ),
            "distortion_IJ": (
                backend.mean(
                    self.distortion_IJ[:, : MS * scale, : NS * scale].reshape(
                        2, MS, scale, NS, scale
                    ),
                    dim=(2, 4),
                )
            ),
            **kwargs,
        }
        return super().reduce(
            scale=scale,
            **kwargs,
        )

    def fluxdensity_to_flux(self):
        self._data = self.data * self.pixel_area_map


class SIPTargetImage(SIPMixin, TargetImage):
    """
    A TargetImage with SIP distortion coefficients.
    This class is used to represent a target image with SIP distortion coefficients.
    It inherits from TargetImage and SIPMixin.
    """

    def model_image(self, upsample: int = 1, pad: int = 0, **kwargs) -> SIPModelImage:
        new_area_map = self.pixel_area_map
        new_distortion_ij = self.distortion_ij
        new_distortion_IJ = self.distortion_IJ
        if upsample > 1:
            new_area_map = (
                backend.upsample2d(new_area_map[None, None], upsample, "nearest")
                .squeeze(0)
                .squeeze(0)
            )
            new_distortion_ij = backend.upsample2d(
                new_distortion_ij[:, None], upsample, "bilinear"
            ).squeeze(1)
            new_distortion_IJ = backend.upsample2d(
                new_distortion_IJ[:, None], upsample, "bilinear"
            ).squeeze(1)
        if pad > 0:
            new_area_map = (
                backend.pad(
                    new_area_map[None, None],
                    (0, 0, 0, 0, pad, pad, pad, pad),
                    mode="replicate",
                )
                .squeeze(0)
                .squeeze(0)
            )
            new_distortion_ij = backend.pad(
                new_distortion_ij[:, None], (0, 0, 0, 0, pad, pad, pad, pad), mode="replicate"
            ).squeeze(1)
            new_distortion_IJ = backend.pad(
                new_distortion_IJ[:, None], (0, 0, 0, 0, pad, pad, pad, pad), mode="replicate"
            ).squeeze(1)
        kwargs = {
            "pixel_area_map": new_area_map,
            "sipA": self.sipA,
            "sipB": self.sipB,
            "sipAP": self.sipAP,
            "sipBP": self.sipBP,
            "distortion_ij": new_distortion_ij,
            "distortion_IJ": new_distortion_IJ,
            "_data": backend.zeros(
                (self.data.shape[0] * upsample + 2 * pad, self.data.shape[1] * upsample + 2 * pad),
                dtype=config.DTYPE,
                device=config.DEVICE,
            ),
            "CD": self.CD.value / upsample,
            "crpix": (self.crpix + 0.5) * upsample + pad - 0.5,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return SIPModelImage(**kwargs)
