from .image_object import BaseImage, Image_List
from .window_object import Window
from ..utils.interpolate import shift_Lanczos_torch
import torch
import numpy as np
from .. import AP_config

__all__ = ["Model_Image", "Model_Image_List"]

class Model_Image(BaseImage):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """
    def __init__(self, pixelscale = None, data = None, window = None, **kwargs):
        assert not (data is None and window is None)
        if data is None:
            data = torch.zeros(tuple(torch.flip(torch.round(window.shape/pixelscale).int(), (0,))), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        super().__init__(data = data, pixelscale = pixelscale, window = window, **kwargs)
        self.to()
        
    def clear_image(self):
        self.data = torch.zeros_like(self.data)

    def shift_origin(self, shift, is_prepadded = True):
        self.window.shift_origin(shift)
        if torch.any(torch.abs(shift/self.pixelscale) > 1):
            raise NotImplementedError("Shifts larger than 1 are currently not handled")
        self.data = shift_Lanczos_torch(self.data, shift[0]/self.pixelscale, shift[1]/self.pixelscale, min(min(self.data.shape), 10), dtype = AP_config.ap_dtype, device = AP_config.ap_device, img_prepadded = is_prepadded)
        
    def replace(self, other, data = None):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any((self.origin + self.shape) < other.origin) or torch.any((other.origin + other.shape) < self.origin):
                return
            other_indices = self.window.get_indices(other)
            self_indices = other.window.get_indices(self)
            if self.data[self_indices].nelement() == 0 or other.data[other_indices].nelement() == 0:
                return
            self.data[self_indices] = other.data[other_indices]
        elif isinstance(other, Window):
            self.data[other.get_indices(self)] = data
        else:
            self.data = other
    
    
class Model_Image_List(Image_List, Model_Image):
    
    def clear_image(self):
        for image in self.image_list:
            image.clear_image()

    def shift_origin(self, shift):
        raise NotImplementedError()

    def replace(self, other, data = None):
        if data is None:
            for image, oth in zip(self.image_list, other):
                image.replace(oth)
        else:
            for image, oth, dat in zip(self.image_list, other, data):
                image.replace(oth, dat)
