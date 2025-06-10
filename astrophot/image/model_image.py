import torch

from .. import AP_config
from .image_object import Image, Image_List
from ..utils.interpolate import shift_Lanczos_torch
from ..errors import InvalidImage

__all__ = ["Model_Image", "Model_Image_List"]


######################################################################
class Model_Image(Image):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """

    def clear_image(self):
        self.data._value = torch.zeros_like(self.data.value)

    def shift(self, shift, is_prepadded=True):
        self.window.shift(shift)
        pix_shift = self.plane_to_pixel_delta(shift)
        if torch.any(torch.abs(pix_shift) > 1):
            raise NotImplementedError("Shifts larger than 1 pixel are currently not handled")
        self.data = shift_Lanczos_torch(
            self.data,
            pix_shift[0],
            pix_shift[1],
            min(min(self.data.shape), 10),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            img_prepadded=is_prepadded,
        )

    def replace(self, other):
        if isinstance(other, Image):
            self_indices = self.get_indices(other)
            other_indices = other.get_indices(self)
            sub_self = self.data._value[self_indices]
            sub_other = other.data._value[other_indices]
            if sub_self.numel() == 0 or sub_other.numel() == 0:
                return
            self.data._value[self_indices] = sub_other
        else:
            raise TypeError(f"Model_Image can only replace with Image objects, not {type(other)}")


######################################################################
class Model_Image_List(Image_List):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not all(isinstance(image, Model_Image) for image in self.image_list):
            raise InvalidImage(
                f"Model_Image_List can only hold Model_Image objects, not {tuple(type(image) for image in self.image_list)}"
            )

    def clear_image(self):
        for image in self.image_list:
            image.clear_image()

    def replace(self, other, data=None):
        if data is None:
            for image, oth in zip(self.image_list, other):
                image.replace(oth)
        else:
            for image, oth, dat in zip(self.image_list, other, data):
                image.replace(oth, dat)
