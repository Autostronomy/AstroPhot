import torch

from .image_object import BaseImage, Image_List
from .. import AP_config

__all__ = ["Jacobian_Image", "Jacobian_Image_List"]

class Jacobian_Image(BaseImage):
    """Image object which represents the evaluation of a jacobian on an image.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # redo everything with 3D tensor
