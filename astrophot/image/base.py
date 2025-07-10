from typing import Optional, Union

import torch
import numpy as np

from ..param import Module
from .. import AP_config
from .window import Window
from . import func


class BaseImage(Module):

    def __init__(
        self,
        *,
        data: Optional[torch.Tensor] = None,
        crpix: Union[torch.Tensor, tuple] = (0.0, 0.0),
        identity: str = None,
        name: Optional[str] = None,
    ) -> None:

        super().__init__(name=name)
        self.data = data  # units: flux
        self.crpix = crpix

        if identity is None:
            self.identity = id(self)
        else:
            self.identity = identity

    @property
    def data(self):
        """The image data, which is a tensor of pixel values."""
        return self._data

    @data.setter
    def data(self, value: Optional[torch.Tensor]):
        """Set the image data. If value is None, the data is initialized to an empty tensor."""
        if value is None:
            self._data = torch.empty((0, 0), dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        else:
            # Transpose since pytorch uses (j, i) indexing when (i, j) is more natural for coordinates
            self._data = torch.transpose(
                torch.as_tensor(value, dtype=AP_config.ap_dtype, device=AP_config.ap_device), 0, 1
            )

    @property
    def crpix(self):
        """The reference pixel coordinates in the image, which is used to convert from pixel coordinates to tangent plane coordinates."""
        return self._crpix

    @crpix.setter
    def crpix(self, value: Union[torch.Tensor, tuple]):
        self._crpix = np.asarray(value, dtype=np.float64)

    @property
    def window(self):
        return Window(window=((0, 0), self.data.shape[:2]), image=self)

    @property
    def shape(self):
        """The shape of the image data."""
        return self.data.shape

    def pixel_center_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, centered on the pixel grid."""
        return func.pixel_center_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_corner_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, with corners at the pixel grid."""
        return func.pixel_corner_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_simpsons_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, with Simpson's rule sampling."""
        return func.pixel_simpsons_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_quad_meshgrid(self, order=3):
        """Get a meshgrid of pixel coordinates in the image, with quadrature sampling."""
        return func.pixel_quad_meshgrid(
            self.shape, AP_config.ap_dtype, AP_config.ap_device, order=order
        )

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        kwargs = {
            "data": torch.transpose(torch.clone(self.data.detach()), 0, 1),
            "crpix": self.crpix,
            "identity": self.identity,
            "name": self.name,
            **kwargs,
        }
        return self.__class__(**kwargs)

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        kwargs = {
            "data": torch.transpose(torch.zeros_like(self.data), 0, 1),
            "crpix": self.crpix,
            "identity": self.identity,
            "name": self.name,
            **kwargs,
        }
        return self.__class__(**kwargs)

    def flatten(self, attribute: str = "data") -> torch.Tensor:
        return getattr(self, attribute).flatten(end_dim=1)

    @torch.no_grad()
    def get_indices(self, other: Window):
        if other.image is self:
            return slice(max(0, other.i_low), min(self.shape[0], other.i_high)), slice(
                max(0, other.j_low), min(self.shape[1], other.j_high)
            )
        shift = np.round(self.crpix - other.crpix).astype(int)
        return slice(
            min(max(0, other.i_low + shift[0]), self.shape[0]),
            max(0, min(other.i_high + shift[0], self.shape[0])),
        ), slice(
            min(max(0, other.j_low + shift[1]), self.shape[1]),
            max(0, min(other.j_high + shift[1], self.shape[1])),
        )

    @torch.no_grad()
    def get_other_indices(self, other: Window):
        if other.image == self:
            shape = other.shape
            return slice(max(0, -other.i_low), min(self.shape[0] - other.i_low, shape[0])), slice(
                max(0, -other.j_low), min(self.shape[1] - other.j_low, shape[1])
            )
        raise ValueError()

    def get_window(self, other: Union[Window, "BaseImage"], indices=None, **kwargs):
        """Get a new image object which is a window of this image
        corresponding to the other image's window. This will return a
        new image object with the same properties as this one, but with
        the data cropped to the other image's window.

        """
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        new_img = self.copy(
            data=self.data[indices],
            crpix=self.crpix - np.array((indices[0].start, indices[1].start)),
            **kwargs,
        )
        return new_img

    def __sub__(self, other):
        if isinstance(other, BaseImage):
            new_img = self[other]
            new_img.data = new_img.data - other[self].data
            return new_img
        else:
            new_img = self.copy()
            new_img.data = new_img.data - other
            return new_img

    def __add__(self, other):
        if isinstance(other, BaseImage):
            new_img = self[other]
            new_img.data = new_img.data + other[self].data
            return new_img
        else:
            new_img = self.copy()
            new_img.data = new_img.data + other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, BaseImage):
            self.data[self.get_indices(other.window)] += other.data[other.get_indices(self.window)]
        else:
            self.data = self.data + other
        return self

    def __isub__(self, other):
        if isinstance(other, BaseImage):
            self.data[self.get_indices(other.window)] -= other.data[other.get_indices(self.window)]
        else:
            self.data = self.data - other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], (BaseImage, Window)):
            return self.get_window(args[0])
        return super().__getitem__(*args)
