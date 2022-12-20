import torch
import numpy as np
from copy import deepcopy
from .window_object import Window
from astropy.io import fits

__all__ = ["BaseImage", "Image_List"]

class BaseImage(object):
    """Core class to represent images. Any image is represented by a data
    matrix, pixelscale, and window in cooridnate space. With this
    information, an image object can undergo arithmatic with other
    image objects while preserving logical image boundaries. The image
    object can also determine coordinate locations for all of its
    pixels (get_coordinate_meshgrid).

    Parameters:
        data: the matrix of pixel values for the image
        pixelscale: the length of one side of a pixel in arcsec/pixel
        window: an AutoProf Window object which defines the spatial cooridnates on the sky
        filename: a filename from which to load the image.
        zeropoint: photometric zero point for converting from pixel flux to magnitude
        note: a note about this image if any
        origin
    """

    def __init__(self, data = None, pixelscale = None, window = None, filename = None, zeropoint = None, note = None, origin = None, center = None, device = None, dtype = torch.float64, **kwargs):
        
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.dtype = dtype
        self._data = None
        
        if filename is not None:
            self.load(filename)
            return
        assert not (pixelscale is None and window is None)
        self.data = data
        self.zeropoint = None if zeropoint is None else torch.as_tensor(zeropoint, dtype = self.dtype, device = self.device)
        self.note = note
        if window is None:
            self.pixelscale = torch.as_tensor(pixelscale, dtype = self.dtype, device = self.device)
            shape = torch.flip(torch.tensor(data.shape, dtype = self.dtype, device = self.device),(0,)) * self.pixelscale
            if origin is None and center is None:
                origin = torch.zeros(2, dtype = self.dtype, device = self.device)
            elif center is None:
                origin = torch.as_tensor(origin, dtype = self.dtype, device = self.device)
            else:
                origin = torch.as_tensor(center, dtype = self.dtype, device = self.device) - shape/2
            self.window = Window(origin = origin, shape = shape, dtype = self.dtype, device = self.device)
        else:
            self.window = window
            self.pixelscale = self.window.shape[0] / self.data.shape[1]
            
            
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
        self._data = data.to(dtype = self.dtype, device = self.device) if isinstance(data, torch.Tensor) else torch.as_tensor(data, dtype = self.dtype, device = self.device)
        
    def copy(self):
        return self.__class__(
            data = torch.clone(self.data),
            device = self.device,
            dtype = self.dtype,
            zeropoint = self.zeropoint,
            note = self.note,
            window = self.window,
        )
    def blank_copy(self):
        return self.__class__(
            data = torch.zeros_like(self.data),
            device = self.device,
            dtype = self.dtype,
            zeropoint = self.zeropoint,
            note = self.note,
            window = self.window,
        )
        
    def get_window(self, window):
        return self.__class__(
            data = self.data[window.get_indices(self)],
            device = self.device,
            dtype = self.dtype,
            pixelscale = self.pixelscale,
            zeropoint = self.zeropoint,
            note = self.note,
            origin = (self.window & window).origin,
        )
    
    def to(self, dtype = None, device = None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        if self._data is not None:
            self._data = self._data.to(dtype = self.dtype, device = self.device)
        self.window.to(dtype = self.dtype, device = self.device)
        return self

    def crop(self, *pixels):
        self.set_data(self.data[pixels[1]:-pixels[1],pixels[0]:-pixels[0]], require_shape = False)
        self.window -= torch.as_tensor(pixels, dtype = self.dtype, device = self.device) * self.pixelscale
        return self

    def flatten(self, attribute = "data"):
        return getattr(self, attribute).reshape(-1)
    
    def get_coordinate_meshgrid_np(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_np(self.pixelscale, x, y)
    def get_coordinate_meshgrid_torch(self, x = 0., y = 0.):
        return self.window.get_coordinate_meshgrid_torch(self.pixelscale, x, y)

    def reduce(self, scale):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self
        
        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        return self.__class__(
            data = self.data[:MS*scale, :NS*scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3)),
            pixelscale = self.pixelscale * scale,
            zeropoint = self.zeropoint,
            note = self.note,
            window = self.window.make_copy(),
        )

    def _save_image_list(self):
        img_header = fits.Header()
        img_header["IMAGE"] = "PRIMARY"
        img_header["PXLSCALE"] = str(self.pixelscale.detach().cpu().item())
        img_header["WINDOW"] = str(self.window.get_state())
        if not self.zeropoint is None:
            img_header["ZEROPNT"] = str(self.zeropoint.detach().cpu().item())
        if not self.note is None:
            img_header["NOTE"] = str(self.note)
        image_list = [fits.PrimaryHDU(self._data.detach().cpu().numpy(), header = img_header)]
        return image_list
    def save(self, filename = None, overwrite = True):
        image_list = self._save_image_list()
        hdul = fits.HDUList(image_list)
        if filename is not None:
            hdul.writeto(filename, overwrite = overwrite)
        return hdul

    def load(self, filename):
        hdul = fits.open(filename)
        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PRIMARY":
                self.set_data(np.array(hdu.data, dtype = np.float64), require_shape = False)
                self.pixelscale = eval(hdu.header.get("PXLSCALE"))
                self.zeropoint = eval(hdu.header.get("ZEROPNT"))
                self.note = hdu.header.get("NOTE")
                self.window = Window(dtype = self.dtype, device = self.device, **eval(hdu.header.get("WINDOW")))
                break
        return hdul
    
    def __sub__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(other.origin + other.shape < self.origin):
                raise IndexError("images have no overlap, cannot subtract!")
            return self.__class__(data = self.data[other.window.get_indices(self)] - other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = (torch.max(self.origin[0], other.origin[0]), torch.max(self.origin[1], other.origin[1])))
        else:
            return self.__class__(data = self.data - other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = self.origin)
        
    def __add__(self, other): # fixme
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(other.origin + other.shape < self.origin):
                return self
            return self.__class__(data = self.data[other.window.get_indices(self)] + other.data[self.window.get_indices(other)],
                            pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = (torch.max(self.origin[0], other.origin[0]), torch.max(self.origin[1], other.origin[1])))
        else:
            return self.__class__(data = self.data + other, pixelscale = self.pixelscale, zeropoint = self.zeropoint, note = self.note, origin = self.origin)

    def __iadd__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] += other.data[self.window.get_indices(other)]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(other.origin + other.shape < self.origin):
                return self
            self.data[other.window.get_indices(self)] -= other.data[self.window.get_indices(other)]
        else:
            self.data -= other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], BaseImage):
            return self.get_window(args[0].window)
        raise ValueError("Unrecognized BaseImage getitem request!")

    def __str__(self):
        return f"image pixelscale: {self.pixelscale} origin: {self.origin}\ndata: {self.data}"

class Image_List(BaseImage):

    def __init__(self, image_list, dtype = torch.float64, device = None):
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.dtype = dtype
        self._window = None
        self.image_list = list(image_list)
        
    @property
    def window(self):
        if self._window is None:
            new_window = self.image_list[0].window.make_copy()
            for image in self.image_list[1:]:
                new_window |= image.window
        return self._window
    @property
    def pixelscale(self):
        return tuple(image.pixelscale for image in self.image_list)
    @property
    def zeropoint(self):
        return tuple(image.zeropoint for image in self.image_list)
    
    @property
    def data(self):
        return tuple(image.data for image in self.image_list)
    @data.setter
    def data(self, data):
        for image, dat in zip(self.image_list, data):
            image.data = dat

    def copy(self):
        return self.__class__(
            tuple(image.copy() for image in self.image_list),
            device = self.device,
            dtype = self.dtype,
        )
    def blank_copy(self):
        return self.__class__(
            tuple(image.blank_copy() for image in self.image_list),
            device = self.device,
            dtype = self.dtype,
        )
        
    def get_window(self, window):
        return self.__class__(
            tuple(image[window] for image in self.image_list),
            device = self.device,
            dtype = self.dtype,
        )
    
    def to(self, dtype = None, device = None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        for image in self.image_list:
            image.to(dtype = self.dtype, device = self.device)
        return self

    def crop(self, *pixels):
        for image in self.image_list:
            image.crop(*pixels)
        self._window = None
        return self
    
    def get_coordinate_meshgrid_np(self, x = 0., y = 0.):
        return tuple(image.get_coordinate_meshgrid_np(x,y) for image in self.image_list)
    def get_coordinate_meshgrid_torch(self, x = 0., y = 0.):
        return tuple(image.get_coordinate_meshgrid_torch(x,y) for image in self.image_list)

    def flatten(self, attribute = "data"):
        return torch.cat(tuple(image.flatten(attribute) for image in self.image_list))

    def reduce(self, scale):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        return self.__class__(
            tuple(image.reduce(scale) for image in self.image_list),
            device = self.device,
            dtype = self.dtype,
        )
    def __sub__(self, other):
        raise NotImplementedError()
    def __add__(self, other):
        raise NotImplementedError()
    def __isub__(self, other):
        raise NotImplementedError()
    def __iadd__(self, other):
        raise NotImplementedError()
    
    def __str__(self):
        return f"image list of:\n" + "\n".join(image.__str__() for image in self.image_list)
    def __iter__(self):
        self.index = 0
        return self
    def __next__(self):
        if self.index >= len(self.image_list):
            raise StopIteration
        img = self.image_list[self.index]
        self.index += 1
        return img
