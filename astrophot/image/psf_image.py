from typing import List, Optional

import torch
import numpy as np

from .image_object import Image
from .jacobian_image import JacobianImage
from .. import config
from .mixins import DataMixin

__all__ = ["PSFImage"]


class PSFImage(DataMixin, Image):
    """Image object which represents a model of PSF (Point Spread Function).

    PSF_Image inherits from the base Image class and represents the model of a point spread function.
    The point spread function characterizes the response of an imaging system to a point source or point object.

    The shape of the PSF data must be odd.

    Attributes:
        data (torch.Tensor): The image data of the PSF.
        identity (str): The identity of the image. Default is None.

    Methods:
        psf_border_int: Calculates and returns the convolution border size of the PSF image in integer format.
        psf_border: Calculates and returns the convolution border size of the PSF image in the units of pixelscale.
        _save_image_list: Saves the image list to the PSF HDU header.
        reduce: Reduces the size of the image using a given scale factor.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"crval": (0, 0), "crpix": (0, 0), "crtan": (0, 0)})
        super().__init__(*args, **kwargs)
        self.crpix = (np.array(self.data.shape, dtype=np.float64) - 1.0) / 2

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        norm = torch.sum(self.data)
        self.data = self.data / norm
        if self.has_weight:
            self.weight = self.weight * norm**2

    @property
    def psf_pad(self):
        return max(self.data.shape) // 2

    def jacobian_image(
        self,
        parameters: Optional[List[str]] = None,
        data: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Construct a blank `Jacobian_Image` object formatted like this current `PSF_Image` object. Mostly used internally.
        """
        if parameters is None:
            data = None
            parameters = []
        elif data is None:
            data = torch.zeros(
                (*self.data.shape, len(parameters)),
                dtype=config.DTYPE,
                device=config.DEVICE,
            )
        kwargs = {
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            **kwargs,
        }
        return JacobianImage(parameters=parameters, data=data, **kwargs)

    def model_image(self, **kwargs):
        """
        Construct a blank `Model_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        kwargs = {
            "data": torch.zeros_like(self.data),
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "identity": self.identity,
            **kwargs,
        }
        return PSFImage(**kwargs)

    @property
    def zeropoint(self):
        return None

    @zeropoint.setter
    def zeropoint(self, value):
        """PSFImage does not support zeropoint."""
        pass

    def plane_to_world(self, x, y):
        raise NotImplementedError(
            "PSFImage does not support plane_to_world conversion. There is no meaningful world position of a PSF image."
        )

    def world_to_plane(self, ra, dec):
        raise NotImplementedError(
            "PSFImage does not support world_to_plane conversion. There is no meaningful world position of a PSF image."
        )
