from .image_object import BaseImage
from .window_object import AP_Window
from autoprof.utils.interpolate import shift_Lanczos
import torch
import numpy as np

class Model_Image(BaseImage):

    def __init__(self, pixelscale = None, data = None, window = None, **kwargs):
        assert not (data is None and window is None)
        if data is None:
            data = torch.zeros(tuple(int(s) for s in np.round(window.shape/pixelscale)))
        super().__init__(data, pixelscale, window, **kwargs)

    def clear_image(self):
        self.data = torch.zeros(self.data.shape)

    def shift_origin(self, shift):
        self.window.shift_origin(shift)
        if np.any(np.abs(shift) > 1):
            raise NotImplementedError("Shifts larger than 1 are currently not handled")
        self.data = shift_Lanczos(self.data, shift[0], shift[1], min(min(self.data.shape), 10))
        
    def replace(self, other, data = None):
        if isinstance(other, BaseImage):
            if not np.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return
            self.data[other.window.get_indices(self)] = other.data[self.window.get_indices(other)]
        elif isinstance(other, AP_Window):
            self.data[other.get_indices(self)] = data
        else:
            self.data = other
    
    def __iadd__(self, other):
        if isinstance(other, BaseImage):
            if not np.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] += other.data[self.window.get_indices(other)]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, BaseImage):
            if not np.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] -= other.data[self.window.get_indices(other)]
        else:
            self.data -= other
        return self

    
