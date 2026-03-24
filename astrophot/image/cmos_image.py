import numpy as np

from .target_image import TargetImage
from .window import Window
from .mixins import CMOSMixin
from .model_image import ModelImage
from ..backend_obj import backend


class CMOSModelImage(CMOSMixin, ModelImage):
    """A ModelImage with CMOS-specific functionality."""


class CMOSTargetImage(CMOSMixin, TargetImage):
    """
    A TargetImage with CMOS-specific functionality.
    This class is used to represent a target image with CMOS-specific features.
    It inherits from TargetImage and CMOSMixin.
    """

    def model_image(self, window: Window, **kwargs) -> CMOSModelImage:
        """Model the image with CMOS-specific features."""
        si, sj = self.get_indices(window)

        kwargs = {
            "subpixel_loc": self.subpixel_loc,
            "subpixel_scale": self.subpixel_scale,
            "_data": backend.zeros_like(self._data[si, sj]),
            "CD": self.CD.value,
            "crpix": self.crpix - np.array((si.start, sj.start)),
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name + "_model",
            **kwargs,
        }
        return CMOSModelImage(**kwargs)
