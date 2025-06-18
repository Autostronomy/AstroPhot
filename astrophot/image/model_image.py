import torch

from .. import AP_config
from .image_object import Image, Image_List
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

    def __init__(self, *args, window=None, upsample=1, pad=0, **kwargs):
        if window is not None:
            kwargs["pixelscale"] = window.image.pixelscale / upsample
            kwargs["crpix"] = (window.crpix + 0.5) * upsample + pad - 0.5
            kwargs["crval"] = window.image.crval
            kwargs["crtan"] = window.image.crtan
            kwargs["data"] = torch.zeros(
                (
                    (window.i_high - window.i_low) * upsample + 2 * pad,
                    (window.j_high - window.j_low) * upsample + 2 * pad,
                ),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            kwargs["zeropoint"] = window.image.zeropoint
        super().__init__(*args, **kwargs)

    def clear_image(self):
        self.data._value = torch.zeros_like(self.data.value)

    def shift_crtan(self, shift):
        # self.data = shift_Lanczos_torch(
        #     self.data,
        #     pix_shift[0],
        #     pix_shift[1],
        #     min(min(self.data.shape), 10),
        #     dtype=AP_config.ap_dtype,
        #     device=AP_config.ap_device,
        #     img_prepadded=is_prepadded,
        # )
        self.crtan._value += shift

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
        if not all(isinstance(image, Model_Image) for image in self.images):
            raise InvalidImage(
                f"Model_Image_List can only hold Model_Image objects, not {tuple(type(image) for image in self.images)}"
            )

    def clear_image(self):
        for image in self.images:
            image.clear_image()

    def replace(self, other, data=None):
        if data is None:
            for image, oth in zip(self.images, other):
                image.replace(oth)
        else:
            for image, oth, dat in zip(self.images, other, data):
                image.replace(oth, dat)
