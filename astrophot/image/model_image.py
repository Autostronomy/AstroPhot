import numpy as np
import torch

from .. import AP_config
from .image_object import Image, ImageList
from ..errors import InvalidImage, SpecificationConflict

__all__ = ["ModelImage", "ModelImageList"]


######################################################################
class ModelImage(Image):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """

    def fluxdensity_to_flux(self):
        self._data = self.data * self.pixel_area


######################################################################
class ModelImageList(ImageList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, ModelImage) for image in self.images):
            raise InvalidImage(
                f"Model_Image_List can only hold Model_Image objects, not {tuple(type(image) for image in self.images)}"
            )

    def clear_image(self):
        for image in self.images:
            image.clear_image()
