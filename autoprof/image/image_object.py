from autoprof.utils.image_operations.load_image import load_fits
import numpy as np
from copy import deepcopy

class AP_Image(np.ndarray):

    def __new__(cls, input_image, pixelscale = None, zeropoint = None, rotation = None, note = None, origin = None, **unused_kwargs):

        obj = np.asarray(input_image).view(cls)
        
        obj.pixelscale = pixelscale
        obj.zeropoint = zeropoint
        obj.rotation = rotation
        obj.origin = np.zeros(2, dtype = int) if origin is None else np.array(origin, dtype = int)
        obj.note = note

        return obj
    
    def __array_finalize__(self, obj):

        if obj is None: return
        
        self.pixelscale = getattr(obj, 'pixelscale', None)
        self.zeropoint = getattr(obj, 'zeropoint', None)
        self.rotation = getattr(obj, 'rotation', None)
        self.origin = getattr(obj, 'origin', np.zeros(2, dtype = int))
        self.note = getattr(obj, 'note', None)

    def clear_image(self):
        self.fill(0)

    def subimage(self, start1 = 0, stop1 = None, start2 = 0, stop2 = None):
        ret = self[start1:stop1,start2:stop2]
        ret.origin = self.origin + np.array([start1, start2])
        return ret

    def add_image(self, other):
        base = other.origin - self.origin
        end = base + other.shape
        self[base[0]:end[0],base[1]:end[1]] += other
