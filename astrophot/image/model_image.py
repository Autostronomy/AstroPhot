import torch
import numpy as np

from .. import AP_config
from .image_object import Image, Image_List
from .window_object import Window
from ..utils.interpolate import shift_Lanczos_torch

__all__ = ["Model_Image", "Model_Image_List"]


######################################################################
class Model_Image(Image):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """

    def __init__(self, pixelscale=None, data=None, window=None, **kwargs):
        assert not (data is None and window is None)
        if data is None:
            data = torch.zeros(
                tuple(window.get_shape_flip(pixelscale)),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        super().__init__(data=data, pixelscale=pixelscale, window=window, **kwargs)
        self.target_identity = kwargs.get("target_identity", None)
        self.to()

    def clear_image(self):
        self.data = torch.zeros_like(self.data)

    def shift_origin(self, shift, is_prepadded=True):
        self.window.shift_origin(shift)
        pix_shift = self.world_to_pixel_delta(shift)
        if torch.any(torch.abs(pix_shift) > 1):
            raise NotImplementedError(
                "Shifts larger than 1 pixel are currently not handled"
            )
        self.data = shift_Lanczos_torch(
            self.data,
            pix_shift[0],
            pix_shift[1],
            min(min(self.data.shape), 10),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            img_prepadded=is_prepadded,
        )

    def get_window(self, window: Window, **kwargs):
        return super().get_window(
            window, target_identity=self.target_identity, **kwargs
        )

    def reduce(self, scale, **kwargs):
        return super().reduce(scale, target_identity=self.target_identity, **kwargs)

    def replace(self, other, data=None):
        if isinstance(other, Image):
            if self.window.overlap_frac(other.window) == 0.0:  # fixme control flow
                return
            other_indices = self.window.get_indices(other)
            self_indices = other.window.get_indices(self)
            if (
                self.data[self_indices].nelement() == 0
                or other.data[other_indices].nelement() == 0
            ):
                return
            self.data[self_indices] = other.data[other_indices]
        elif isinstance(other, Window):
            self.data[other.get_indices(self)] = data
        else:
            self.data = other


######################################################################
class Model_Image_List(Image_List, Model_Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert all(
            isinstance(image, Model_Image) for image in self.image_list
        ), f"Model_Image_List can only hold Model_Image objects, not {tuple(type(image) for image in self.image_list)}"

    def clear_image(self):
        for image in self.image_list:
            image.clear_image()

    def shift_origin(self, shift):
        raise NotImplementedError()

    def replace(self, other, data=None):
        if data is None:
            for image, oth in zip(self.image_list, other):
                image.replace(oth)
        else:
            for image, oth, dat in zip(self.image_list, other, data):
                image.replace(oth, dat)

    @property
    def target_identity(self):
        targets = tuple(image.target_identity for image in self.image_list)
        if any(tar_id is None for tar_id in targets):
            return None
        return targets

    def __isub__(self, other):
        if isinstance(other, Model_Image_List):
            for other_image, zip_self_image in zip(other.image_list, self.image_list):
                if other_image.target_identity is None or self.target_identity is None:
                    zip_self_image -= other_image
                    continue
                for self_image in self.image_list:
                    if other_image.target_identity == self_image.target_identity:
                        self_image -= other_image
                        break
                else:
                    self.image_list.append(other_image)
        elif isinstance(other, Model_Image):
            if other.target_identity is None or zip_self_image.target_identity is None:
                zip_self_image -= other_image
            else:
                for self_image in self.image_list:
                    if other.target_identity == self_image.target_identity:
                        self_image -= other
                        break
                else:
                    self.image_list.append(other)
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image -= other_image
        return self

    def __iadd__(self, other):
        if isinstance(other, Model_Image_List):
            for other_image, zip_self_image in zip(other.image_list, self.image_list):
                if other_image.target_identity is None or self.target_identity is None:
                    zip_self_image += other_image
                    continue
                for self_image in self.image_list:
                    if other_image.target_identity == self_image.target_identity:
                        self_image += other_image
                        break
                else:
                    self.image_list.append(other_image)
        elif isinstance(other, Model_Image):
            if other.target_identity is None or self.target_identity is None:
                for self_image in self.image_list:
                    self_image += other
            else:
                for self_image in self.image_list:
                    if other.target_identity == self_image.target_identity:
                        self_image += other
                        break
                else:
                    self.image_list.append(other)
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image += other_image
        return self
