from .target_image import TargetImage
from .mixins import CMOSMixin
from .model_image import ModelImage
from ..backend_obj import backend
from .. import config


class CMOSModelImage(CMOSMixin, ModelImage):
    """A ModelImage with CMOS-specific functionality."""

    def fluxdensity_to_flux(self):
        # CMOS pixels only sensitive in sub area, so scale the flux density
        self._data = self.data * self.pixel_area * self.subpixel_scale**2


class CMOSTargetImage(CMOSMixin, TargetImage):
    """
    A TargetImage with CMOS-specific functionality.
    This class is used to represent a target image with CMOS-specific features.
    It inherits from TargetImage and CMOSMixin.
    """

    def model_image(self, upsample: int = 1, pad: int = 0, **kwargs) -> CMOSModelImage:
        """Model the image with CMOS-specific features."""
        if upsample > 1 or pad > 0:
            raise NotImplementedError("Upsampling and padding are not implemented for CMOS images.")

        kwargs = {
            "subpixel_loc": self.subpixel_loc,
            "subpixel_scale": self.subpixel_scale,
            "_data": backend.zeros(self.data.shape[:2], dtype=config.DTYPE, device=config.DEVICE),
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return CMOSModelImage(**kwargs)
