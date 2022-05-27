import numpy as np
from copy import deepcopy


class AP_Image(object):

    def __init__(self, data, pixelscale = None, zeropoint = None, rotation = None, note = None, origin = None, **kwargs):

        self.data = data
        self.pixelscale = pixelscale
        self.zeropoint = zeropoint
        self.rotation = rotation
        self.note = note
        self.origin = np.zeros(2,dtype=int) if origin is None else np.array(origin)
        self.shape = data.shape
        
    def clear_image(self):
        self.data.fill(0)

    def subimage(self, low_x = 0, high_x = None, low_y = 0, high_y = None):
        return AP_Image(self.data[low_x:high_x, low_y:high_y], pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = self.origin + np.array([low_x,low_y]))

    def get_area(self, origin, shape):
        return AP_Image(self.data[origin[0] - self.origin[0]:origin[0] + shape[0] - self.origin[0],
                                  origin[1] - self.origin[1]:origin[1] + shape[1] - self.origin[1]],
                        pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = origin)
    
    def get_image_area(self, image):
        return self.get_area(image.origin, image.shape)
    
    def __iadd__(self, other):
        if isinstance(other, AP_Image):
            if np.any(self.origin + self.data.shape < other.origin) or np.any(other.origin + other.data.shape < self.origin):
                return self
            self_base = np.clip(other.origin - self.origin, a_min = 0, a_max = None)
            self_end = other.origin - self.origin + np.array(other.data.shape)
            self_end[0] = min(self_end[0],self.data.shape[0])
            self_end[1] = min(self_end[1],self.data.shape[1])
            other_base = np.clip(self.origin - other.origin, a_min = 0, a_max = None)
            other_end = self.origin - other.origin + np.array(self.data.shape)
            other_end[0] = min(other_end[0],other.data.shape[0])
            other_end[1] = min(other_end[1],other.data.shape[1])
            self.data[self_base[0]:self_end[0],self_base[1]:self_end[1]] += other.data[other_base[0]:other_end[0],other_base[1]:other_end[1]]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, AP_Image):
            base = other.origin - self.origin
            end = base + other.data.shape
            self.data[base[0]:end[0],base[1]:end[1]] -= other.data
        else:
            self.data -= other
        return self

    def __sub__(self, other):
        if isinstance(other, AP_Image):
            base = other.origin - self.origin
            end = base + other.data.shape
            return AP_Image(self.data[base[0]:end[0],base[1]:end[1]] - other.data, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = other.origin)
        else:
            return AP_Image(self.data - other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = self.origin)
        
    def __add__(self, other):
        if isinstance(other, AP_Image):
            base = other.origin - self.origin
            end = base + other.data.shape
            return AP_Image(self.data[base[0]:end[0],base[1]:end[1]] + other.data, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = other.origin)
        else:
            return AP_Image(self.data + other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, rotation = self.rotation, note = self.note, origin = self.origin)
