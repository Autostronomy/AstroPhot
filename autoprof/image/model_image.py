import torch
import numpy as np

from .. import AP_config
from .image_object import Image, Image_List, Image_Batch
from .window_object import Window
from .window_batch import Window_Batch
from .image_header import Image_Header_Batch
from ..utils.interpolate import shift_Lanczos_torch

__all__ = ["Model_Image", "Model_Image_List", "Model_Image_Batch"]

######################################################################
class Model_Image(Image):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """
    subclasses = {}

    def __init_subclass__(cls):
        if hasattr(cls, "subname"):
            Model_Image.subclasses[cls.subname] = cls
        
    def __new__(cls, *args, **kwargs):
        if (isinstance(kwargs.get("origin", None), torch.Tensor) and kwargs["origin"].dim() == 2) or isinstance(kwargs.get("header", None), Image_Header_Batch) or isinstance(kwargs.get("window", None), Window_Batch):
            return super().__new__(Model_Image.subclasses["batch"])
        if kwargs.get("image_list", None) is not None:
            return super().__new__(Model_Image.subclasses["list"])
        return super().__new__(cls)
    
    def __init__(self, pixelscale=None, data=None, window=None, **kwargs):
        assert not (data is None and window is None)
        if data is None:
            data_shape = tuple(torch.flip(torch.round(window.shape / pixelscale).int(), (0,)))
            if window.origin.dim() == 2:
                data_shape = (window.origin.shape[0],) + data_shape 
            data = torch.zeros(
                data_shape,
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
        if torch.any(torch.abs(shift / self.pixelscale) > 1):
            raise NotImplementedError("Shifts larger than 1 are currently not handled")
        self.data = shift_Lanczos_torch(
            self.data,
            shift[0] / self.pixelscale,
            shift[1] / self.pixelscale,
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
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any((self.origin + self.shape) < other.origin) or torch.any(
                (other.origin + other.shape) < self.origin
            ):
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
    subname = "list"
    
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

######################################################################
class Model_Image_Batch(Model_Image, Image_Batch):
    """Image object which represents the sampling of a model at the given
    coordinates of the image. Extra arithmetic operations are
    available which can update model values in the image. The whole
    model can be shifted by less than a pixel to account for sub-pixel
    accuracy.

    """
    subname = "batch"

    def __init__(self, pixelscale=None, data=None, window=None, **kwargs):
        assert not (data is None and window is None)
        if data is None:
            data_shape = tuple(torch.flip(torch.round(window.shape / pixelscale).int(), (0,)))
            data_shape = (window.origin.shape[0],) + data_shape 
            data = torch.zeros(
                data_shape,
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        super().__init__(data=data, pixelscale=pixelscale, window=window, **kwargs)
        self.target_identity = kwargs.get("target_identity", None)
        self.to()

    def squish(self):
        squish_window = self.window.squish()
        squish_image = Model_Image(
            data = torch.zeros(*squish_window.get_shape_flip(self.pixelscale)),
            window = squish_window,
            pixelscale = self.pixelscale,
        )
        squish_image += self
        return squish_image

    def shift_origin(self, shift, is_prepadded=True):
        self.window.shift_origin(shift)
        if torch.any(torch.abs(shift / self.pixelscale) > 1):
            raise NotImplementedError("Shifts larger than 1 are currently not handled")
        self.data = shift_Lanczos_torch( # fixme
            self.data,
            shift[0] / self.pixelscale,
            shift[1] / self.pixelscale,
            min(min(self.data.shape[1:]), 10),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            img_prepadded=is_prepadded,
        )

    def replace(self, other, data=None):
        if isinstance(other, Image_Batch):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any((self.origin + self.shape) < other.origin) or torch.any(
                (other.origin + other.shape) < self.origin
            ):
                return
            other_indices = self.window.get_indices(other)
            self_indices = other.window.get_indices(self)
            if (
                self.data[self_indices].nelement() == 0
                or other.data[other_indices].nelement() == 0
            ):
                return
            for i, oi, si in zip(range(len(self_indices)), other_indices, self_indices):
                self.data[i][self_indices] = other.data[i][other_indices]
        elif isinstance(other, Window_Batch):
            self_indices = other.get_indices(self)
            for i, si in enumerate(self_indices): 
                self.data[i][si] = data[i]
        else:
            self.data = other

    def crop(self, pixels):
        if len(pixels) == 1:  # same crop in all dimension
            self.set_data(
                self.data[:,
                    pixels[0] : self.data.shape[0] - pixels[0],
                    pixels[0] : self.data.shape[1] - pixels[0],
                ],
                require_shape=False,
            )
        elif len(pixels) == 2:  # different crop in each dimension
            self.set_data(
                self.data[:,
                    pixels[1] : self.data.shape[0] - pixels[1],
                    pixels[0] : self.data.shape[1] - pixels[0],
                ],
                require_shape=False,
            )
        elif len(pixels) == 4:  # different crop on all sides
            self.set_data(
                self.data[:,
                    pixels[2] : self.data.shape[0] - pixels[3],
                    pixels[0] : self.data.shape[1] - pixels[1],
                ],
                require_shape=False,
            )
        self.header = self.header.crop(pixels)
        return self

    def flatten(self, attribute: str = "data") -> torch.Tensor:
        return getattr(self, attribute).reshape(self.data.shape[0], -1)

    def reduce(self, scale: int, **kwargs):
        """This operation will downsample an image by the factor given. If
        scale = 2 then 2x2 blocks of pixels will be summed together to
        form individual larger pixels. A new image object will be
        returned with the appropriate pixelscale and data tensor. Note
        that the window does not change in this operation since the
        pixels are condensed, but the pixel size is increased
        correspondingly.

        Parameters:
            scale: factor by which to condense the image pixels. Each scale X scale region will be summed [int]

        """
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        MS = self.data.shape[1] // scale
        NS = self.data.shape[2] // scale
        return self.__class__(
            data=self.data[:, : MS * scale, : NS * scale]
            .reshape(self.data.shape[0], MS, scale, NS, scale)
            .sum(axis=(2, 4)),
            header=self.header.reduce(scale, **kwargs),
            **kwargs,
        )
            
    def __sub__(self, other):
        if isinstance(other, Image_Batch):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                raise IndexError("images have no overlap, cannot subtract!")
            new_img = self[other.window]
            other_indices = self.window.get_indices(other)
            for i ,oi in zip(range(len(other_indices)), other_indices):
                new_img.data[i] -= other.data[i][oi]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            new_img = self[other.window].copy()
            other_indices = self.window.get_indices(other)
            for i ,oi in zip(range(len(other_indices)), other_indices):
                new_img.data[i] += other.data[i][oi]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data += other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image_Batch):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            self_indices = other.window.get_indices(self)
            other_indices = self.window.get_indices(other)
            for i, si, oi in zip(range(len(self_indices)), self_indices, other_indices):
                self.data[i][si] += other.data[i][oi]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            self_indices = other.window.get_indices(self)
            other_indices = self.window.get_indices(other)
            for i, si, oi in zip(range(len(self_indices)), self_indices, other_indices):
                self.data[i][si] -= other.data[i][oi]
        else:
            self.data -= other
        return self
