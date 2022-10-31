import torch
import numpy as np
from copy import deepcopy
from .window_object import AP_Window

class BaseImage(object):
    """Core class to represent images. Any image is represented by a data
    matrix, pixelscale, and window in cooridnate space. With this
    information, an image object can undergo arithmatic with other
    image objects while preserving logical image boundaries. The image
    object can also determine coordinate locations for all of its
    pixels (get_coordinate_meshgrid).

    """

    def __init__(self, data, pixelscale = None, window = None, zeropoint = None, note = None, origin = None, center = None, **kwargs):
        assert not (pixelscale is None and window is None)
        self._data = None
        self.data = data #.to(dtype = torch.float64) if isinstance(data, torch.Tensor) else torch.tensor(data, dtype = torch.float64)
        self.zeropoint = zeropoint
        self.note = note
        if window is None:
            self.pixelscale = pixelscale
            shape = np.flip(np.array(data.shape)) * self.pixelscale
            if origin is None and center is None:
                origin = np.zeros(2)
            elif center is None:
                origin = np.array(origin)
            else:
                origin = np.array(center) - shape/2
            self.window = AP_Window(origin = origin, shape = shape)
        else:
            self.window = window
            self.pixelscale = float(self.window.shape[0]) / float(self.data.shape[1])
            
            
    @property
    def origin(self):
        return self.window.origin
    @property
    def shape(self):
        return self.window.shape
    @property
    def center(self):
        return self.window.center

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data):
        self.set_data(data)
        
    def set_data(self, data, require_shape = True):
        if self._data is not None and require_shape:
            assert data.shape == self._data.shape
        self._data = data.to(dtype = torch.float64) if isinstance(data, torch.Tensor) else torch.tensor(data, dtype = torch.float64)
        
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
        self.set_data(self.data[pixels[1]:-pixels[1],pixels[0]:-pixels[0]], require_shape = False)
        self.window -= tuple(np.array(pixels) * self.pixelscale)

    def get_coordinate_meshgrid_np(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_np(self.pixelscale, x, y)
    def get_coordinate_meshgrid_torch(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_torch(self.pixelscale, x, y)

    def reduce(self, scale):
        assert isinstance(scale, int)
        assert scale > 1

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        return self.__class__(
            data = self.data.detach().numpy()[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            pixelscale = self.pixelscale * scale,
            zeropoint = self.zeropoint,
            note = self.note,
            origin = self.origin,
        )
    
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
