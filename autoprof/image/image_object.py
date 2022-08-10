import numpy as np
from copy import deepcopy
from .window_object import AP_Window

class AP_Image(object):

    def __init__(self, data, pixelscale, zeropoint = None, rotation = None, note = None, origin = None, window = None, **kwargs):

        self.data = data
        self.pixelscale = pixelscale
        self.zeropoint = zeropoint
        self.rotation = rotation
        self.note = note
        if window is None:
            self.origin = np.zeros(2) if origin is None else np.array(origin)
            self.shape = np.array(data.shape) * self.pixelscale
            self.window = AP_Window(origin = self.origin, shape = self.shape)
        else:
            self.window = window
            self.origin = self.window.origin
            self.shape = self.window.shape
            
    def clear_image(self):
        self.data.fill(0)

    def blank_copy(self):
        return AP_Image(
            np.zeros(self.data.shape),
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            rotation = self.rotation,
            note = self.note,
            origin = self.origin,
        )
        
    def get_window(self, window):
        return AP_Image(
            self.data[window.get_indices(self)],
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            rotation = self.rotation,
            note = self.note,
            origin = (max(self.origin[0], window.origin[0]),
                      max(self.origin[1], window.origin[1]))
        )

    def get_coordinate_meshgrid(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid(self.pixelscale, x, y)

    def replace(self, other, data = None):
        if isinstance(other, AP_Image):
            if self.pixelscale != other.pixelscale:
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return
            self.data[other.window.get_indices(self)] = other.data[self.window.get_indices(other)]
        elif isinstance(other, AP_Window):
            self.data[other.get_indices(self)] = data
        else:
            self.data = other
    
    def __iadd__(self, other):
        if isinstance(other, AP_Image):
            if self.pixelscale != other.pixelscale:
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] += other.data[self.window.get_indices(other)]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, AP_Image):
            if self.pixelscale != other.pixelscale:
                raise IndexError("Cannot subtract images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] -= other.data[self.window.get_indices(other)]
        else:
            self.data -= other
        return self

    def __sub__(self, other):
        if isinstance(other, AP_Image):
            if self.pixelscale != other.pixelscale:
                raise IndexError("Cannot subtract images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                raise IndexError("images have no overlap, cannot subtract!")
            return AP_Image(self.data[other.window.get_indices(self)] - other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = (max(self.origin[0], other.origin[0]), max(self.origin[1], other.origin[1])))
        else:
            return AP_Image(self.data - other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = self.origin)
        
    def __add__(self, other): # fixme
        if isinstance(other, AP_Image):
            if self.pixelscale != other.pixelscale:
                raise IndexError("Cannot add images with different pixelscale!")
            if np.any(self.origin + self.shape < other.origin) or np.any(other.origin + other.shape < self.origin):
                return self
            return AP_Image(self.data[other.window.get_indices(self)] + other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = (max(self.origin[0], other.origin[0]), max(self.origin[1], other.origin[1])))
        else:
            return AP_Image(self.data + other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = self.origin)

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], AP_Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], AP_Image):
            return self.get_window(args[0].window)
        raise ValueError("Unrecognized AP_Image getitem request!")

    def __str__(self):
        return f"image pixelscale: {self.pixelscale} data: {self.data}"
