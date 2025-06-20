from typing import List, Optional

import torch
import numpy as np

from .image_object import Image
from .model_image import ModelImage
from .jacobian_image import JacobianImage
from .. import AP_config

__all__ = ["PSFImage"]


class PSFImage(Image):
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

    has_mask = False
    has_variance = False

    def __init__(self, *args, **kwargs):
        kwargs.update({"crval": (0, 0), "crpix": (0, 0), "crtan": (0, 0)})
        super().__init__(*args, **kwargs)
        self.crpix = (np.array(self.data.shape, dtype=float) - 1.0) / 2

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        self.data._value /= torch.sum(self.data.value)

    @property
    def mask(self):
        return torch.zeros_like(self.data.value, dtype=bool)

    @property
    def psf_border_int(self):
        """Calculates and returns the border size of the PSF image in integer
        format. This is the border used for padding before convolution.

        Returns:
            torch.Tensor: The border size of the PSF image in integer format.

        """
        return torch.tensor(
            self.data.shape,
            dtype=torch.int32,
            device=AP_config.ap_device,
        )

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
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        kwargs = {
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
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
            "data": torch.zeros_like(self.data.value),
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            **kwargs,
        }
        return ModelImage(**kwargs)
