import torch
import numpy as np
from copy import deepcopy
from .window_object import AP_Window

class BaseImage(object):# refactor to put pixelscale first, then allow pixelscale plus window as initialization

    def __init__(self, data, pixelscale = None, window = None, zeropoint = None, note = None, origin = None, **kwargs):

        assert not (pixelscale is None and window is None)
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype = torch.float32)
        self.zeropoint = zeropoint
        self.note = note
        if window is None:
            self.pixelscale = pixelscale
            origin = np.zeros(2) if origin is None else np.array(origin)
            shape = np.array(data.shape) * self.pixelscale
            self.window = AP_Window(origin = origin, shape = shape)
        else:
            self.window = window
            self.pixelscale = self.window.shape[0] / self.data.shape[0]
            
    @property
    def origin(self):
        return self.window.origin
    @property
    def shape(self):
        return self.window.shape
    @property
    def center(self):
        return self.window.center
            
    def blank_copy(self):
        return self.__class__(
            data = torch.zeros(self.data.shape, dtype = torch.float32),
            zeropoint = self.zeropoint,
            note = self.note,
            window = self.window,
        )
        
    def get_window(self, window):
        return self.__class__(
            data = self.data[window.get_indices(self)],
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            note = self.note,
            origin = (max(self.origin[0], window.origin[0]),
                      max(self.origin[1], window.origin[1]))
        )

    def crop(self, *pixels):
        self.data = self.data[pixels[0]:-pixels[0],pixels[1]:-pixels[1]]
        self.window -= tuple(np.array(pixels) * self.pixelscale)

    def get_coordinate_meshgrid_np(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_np(self.pixelscale, x, y)
    def get_coordinate_meshgrid_torch(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_torch(self.pixelscale, x, y)

    def __sub__(self, other):
        if isinstance(other, BaseImage):
            if not np.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                raise IndexError("images have no overlap, cannot subtract!")
            return self.__class__(data = self.data[other.window.get_indices(self)] - other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = (max(self.origin[0], other.origin[0]), max(self.origin[1], other.origin[1])))
        else:
            return self.__class__(data = self.data - other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = self.origin)
        
    def __add__(self, other): # fixme
        if isinstance(other, BaseImage):
            if not np.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            return self.__class__(data = self.data[other.window.get_indices(self)] + other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = (max(self.origin[0], other.origin[0]), max(self.origin[1], other.origin[1])))
        else:
            return self.__class__(data = self.data + other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = self.origin)

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], AP_Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], BaseImage):
            return self.get_window(args[0].window)
        raise ValueError("Unrecognized BaseImage getitem request!")

    def __str__(self):
        return f"image pixelscale: {self.pixelscale} origin: {self.origin}\ndata: {self.data}"
