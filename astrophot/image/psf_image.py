from typing import List, Optional, Union

import torch
import numpy as np
from torch.nn.functional import avg_pool2d

from .image_object import Image, Image_List
from .image_header import Image_Header
from .model_image import Model_Image
from .jacobian_image import Jacobian_Image
from astropy.io import fits
from .. import AP_config
from ..errors import SpecificationConflict, InvalidData

__all__ = ["PSF_Image"]

class PSF_Image(Image):
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
        """
        Initializes the PSF_Image class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                band (str, optional): The band of the image. Default is None.
        """
        super().__init__(*args, **kwargs)
        self.window.reference_radec = (0,0)
        self.window.reference_planexy = (0,0)
        self.window.reference_imageij = np.flip(np.array(self.data.shape, dtype = float) - 1.) / 2
        self.window.reference_imagexy = (0,0)

    def set_data(
        self, data: Union[torch.Tensor, np.ndarray], require_shape: bool = True
    ):
        super().set_data(data = data, require_shape = require_shape)
        
        if torch.any(
            (torch.tensor(self.data.shape) % 2) != 1
        ):
            raise SpecificationConflict(f"psf must have odd shape, not {self.data.shape}")
        if torch.any(self.data < 0):
            AP_config.ap_logger.warning("psf data should be non-negative")

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        self.data /= torch.sum(self.data)

    @property
    def psf_border_int(self):
        """Calculates and returns the border size of the PSF image in integer
        format. This is the border used for padding before convolution.

        Returns:
            torch.Tensor: The border size of the PSF image in integer format.

        """
        return torch.ceil(
            (
                1
                + torch.flip(
                    torch.tensor(
                        self.data.shape,
                        dtype=AP_config.ap_dtype,
                        device=AP_config.ap_device,
                    ),
                    (0,),
                )
            )
            / 2
        ).int()

    def _save_image_list(self, image_list):
        """Saves the image list to the PSF HDU header.

        Args:
            image_list (list): The list of images to be saved.
            psf_header (astropy.io.fits.Header): The header of the PSF HDU.
        """
        img_header = self.header._save_image_list()
        img_header["IMAGE"] = "PSF"
        image_list.append(
            fits.ImageHDU(self.data.detach().cpu().numpy(), header=img_header)
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
        return Jacobian_Image(
            parameters=parameters,
            target_identity=self.identity,
            data=data,
            header=self.header,
            **kwargs,
        )
    
    def model_image(self, data: Optional[torch.Tensor] = None, **kwargs):
        """
        Construct a blank `Model_Image` object formatted like this current `Target_Image` object. Mostly used internally.
        """
        return Model_Image(
            data=torch.zeros_like(self.data) if data is None else data,
            header=self.header,
            target_identity=self.identity,
            **kwargs,
        )

    def expand(self, padding):
        raise NotImplementedError("expand not available for PSF_Image")
    
    def get_fits_state(self):
        states = [{}]
        states[0]["DATA"] = self.data.detach().cpu().numpy()
        states[0]["HEADER"] = self.header.get_fits_state()
        states[0]["HEADER"]["IMAGE"] = "PSF"
        return states

    def set_fits_state(self, states):
        for state in states:
            if state["HEADER"]["IMAGE"] == "PSF":
                self.set_data(np.array(state["DATA"], dtype=np.float64), require_shape=False)
                self.header = Image_Header(fits_state = state["HEADER"])
                break
