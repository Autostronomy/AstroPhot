from .image_object import BaseImage
from .window_object import Window
from autoprof.utils.interpolate import shift_Lanczos_torch
import torch
import numpy as np

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
            data = torch.zeros(tuple(torch.flip(torch.round(window.shape/pixelscale).int(), (0,))), dtype = kwargs.get("dtype", torch.float64), device = kwargs.get("device", "cpu"))
        super().__init__(data = data, pixelscale = pixelscale, window = window, **kwargs)
        
    def clear_image(self):
        self.data = torch.zeros_like(self.data)

    def shift_origin(self, shift):
        self.window.shift_origin(shift)
        if torch.any(torch.abs(shift/self.pixelscale) > 1):
            raise NotImplementedError("Shifts larger than 1 are currently not handled")
        self.data = shift_Lanczos_torch(self.data, shift[0]/self.pixelscale, shift[1]/self.pixelscale, min(min(self.data.shape), 10), dtype = self.dtype, device = self.device)
        
    def replace(self, other, data = None):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any((self.origin + self.shape) < other.origin) or torch.any((other.origin + other.shape) < self.origin):
                return
            other_indices = self.window.get_indices(other)
            self_indices = other.window.get_indices(self)
            self.data[self_indices] = other.data[other_indices]
        elif isinstance(other, Window):
            self.data[other.get_indices(self)] = data
        else:
            self.data = other
    
    
