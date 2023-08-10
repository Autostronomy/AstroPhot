from typing import List, Optional

import torch
import numpy as np
from torch.nn.functional import avg_pool2d

from .image_object import Image, Image_List
from astropy.io import fits
from .. import AP_config

__all__ = ["PSF_Image"]


class PSF_Image(Image):
    """Image object which represents a model of PSF (Point Spread Function).

    PSF_Image inherits from the base Image class and represents the model of a point spread function.
    The point spread function characterizes the response of an imaging system to a point source or point object.

    The shape of the PSF data must be odd.

    Attributes:
        data (torch.Tensor): The image data of the PSF.
        psf_upscale (torch.Tensor): Upscaling factor of the PSF. Default is 1.
        identity (str): The identity of the image. Default is None.

    Methods:
        psf_border_int: Calculates and returns the convolution border size of the PSF image in integer format.
        psf_border: Calculates and returns the convolution border size of the PSF image in the units of pixelscale.
        _save_image_list: Saves the image list to the PSF HDU header.
        reduce: Reduces the size of the image using a given scale factor.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the PSF_Image class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                psf_upscale (int, optional): Upscaling factor of the PSF. Default is 1.
                band (str, optional): The band of the image. Default is None.
        """
        self.psf_upscale = torch.as_tensor(
            kwargs.get("psf_upscale", 1), dtype=torch.int32, device=AP_config.ap_device
        )
        super().__init__(*args, **kwargs)
        assert torch.all(
            (torch.tensor(self.data.shape) % 2) == 1
        ), "psf must have odd shape"

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
                / self.psf_upscale
            )
            / 2
        ).int()

    @property
    def psf_border(self):
        """Calculates and returns the border size of the PSF image in the
        units of pixelscale. This is the border used for padding
        before convolution.

        Returns:
            torch.Tensor: The border size of the PSF image in the units of pixelscale.

        """
        return self.window.world_to_cartesian(
            self.pixel_to_world_delta(self.psf_border_int.to(dtype=AP_config.ap_dtype))
        )

    def _save_image_list(self, image_list, psf_header):
        """Saves the image list to the PSF HDU header.

        Args:
            image_list (list): The list of images to be saved.
            psf_header (astropy.io.fits.Header): The header of the PSF HDU.
        """
        psf_header["IMAGE"] = "PSF"
        psf_header["UPSCALE"] = int(self.psf_upscale.detach().cpu().item())
        image_list.append(
            fits.ImageHDU(self.data.detach().cpu().numpy(), header=psf_header)
        )

    def copy(self, **kwargs):
        """Creates a copy of the PSF_Image instance.

        This method generates a copy of the PSF_Image object while maintaining the 'psf_upscale' and 'band' properties.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            PSF_Image: A copy of the current PSF_Image instance.
        """
        return super().copy(
            psf_upscale=self.psf_upscale,
        )

    def blank_copy(self, **kwargs):
        """Creates a blank copy of the PSF_Image instance.

        This method generates a blank copy of the PSF_Image object while maintaining the 'psf_upscale' and 'band' properties.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            PSF_Image: A blank copy of the current PSF_Image instance with the same properties but no data.
        """
        return super().blank_copy(
            psf_upscale=self.psf_upscale,
        )

    def get_window(self, **kwargs):
        """Returns the window of the PSF_Image instance.

        This method returns the window of the PSF_Image object while maintaining the 'psf_upscale' and 'band' properties.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Window: The window associated with the PSF_Image instance.
        """
        return super().get_window(
            psf_upscale=self.psf_upscale,
        )

    def to(self, dtype=None, device=None):
        """Transfers the PSF_Image instance to the specified device and modifies its data type.

        This method changes the data type of the PSF_Image object and moves it to a specified device.
        If no device is provided, it defaults to 'AP_config.ap_device'.

        Args:
            dtype (torch.dtype, optional): The desired data type to which the PSF_Image instance should be converted.
            device (torch.device, optional): The desired device to which the PSF_Image instance should be moved.
        """
        if device is None:
            device = AP_config.ap_device
        self.psf_upscale = self.psf_upscale.to(device=device)
        super().to(dtype=dtype, device=device)

    def reduce(self, scale, **kwargs):
        """Reduces the size of the image using a given scale factor.

        This method is used to perform a reduction in the size of the PSF image. The new upscaling factor
        is calculated by dividing the existing upscaling factor with the provided scale factor.

        Args:
            scale (float): The scale factor by which the size of the PSF image needs to be reduced.
            **kwargs: Arbitrary keyword arguments. This can be used to pass additional parameters required by the method.

        Returns:
            PSF_Image: A new instance of PSF_Image class with the reduced image size.
        """
        return super().reduce(scale, psf_upscale=self.psf_upscale / scale, **kwargs)

    def expand(self, padding):
        raise NotImplementedError("expand not available for PSF_Image")
