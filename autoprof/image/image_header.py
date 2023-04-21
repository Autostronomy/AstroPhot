from typing import Optional, Union, Any, Sequence, Tuple
from copy import deepcopy

import torch
from torch.nn.functional import pad
import numpy as np
from astropy.io import fits

from .window_object import Window, Window_List
from .. import AP_config

__all__ = ["Image_Header"]

class Image_Header(object):

    def __init__(
        self,
        data_shape: Optional[torch.Tensor] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        window: Optional[Window] = None,
        filename: Optional[str] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        note: Optional[str] = None,
        origin: Optional[Sequence] = None,
        center: Optional[Sequence] = None,
        _identity: str = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of the APImage class.

        Parameters:
        -----------
        pixelscale : float or None, optional
            The physical scale of the pixels in the image, in units of arcseconds. Default is None.
        window : Window or None, optional
            A Window object defining the area of the image to use. Default is None.
        filename : str or None, optional
            The name of a file containing the image data. Default is None.
        zeropoint : float or None, optional
            The image's zeropoint, used for flux calibration. Default is None.
        note : str or None, optional
            A note describing the image. Default is None.
        origin : numpy.ndarray or None, optional
            The origin of the image in the coordinate system, as a 1D array of length 2. Default is None.
        center : numpy.ndarray or None, optional
            The center of the image in the coordinate system, as a 1D array of length 2. Default is None.

        Returns:
        --------
        None
        """
        # Record identity
        if _identity is None:
            self.identity = str(id(self))
        else:
            self.identity = _identity

        if filename is not None:
            self.load(filename)
            return

        assert not (pixelscale is None and window is None)

        # set Zeropoint
        if zeropoint is None:
            self.zeropoint = None
        else:
            self.zeropoint = torch.as_tensor(
                zeropoint, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

        # set a note for the image
        self.note = note

        # Set Window
        if window is None:
            # If window is not provided, create one based on pixelscale and data shape
            assert (
                pixelscale is not None
            ), "pixelscale cannot be None if window is not provided"

            self.pixelscale = torch.as_tensor(
                pixelscale, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            shape = (
                torch.flip(
                    torch.tensor(
                        data_shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                    ),
                    (0,),
                )
                * self.pixelscale
            )
            if origin is None and center is None:
                origin = torch.zeros(
                    2, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            elif center is None:
                origin = torch.as_tensor(
                    origin, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            else:
                origin = (
                    torch.as_tensor(
                        center, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                    )
                    - shape / 2
                )

            self.window = Window(origin=origin, shape=shape)
        else:
            # When The Window object is provided
            self.window = window
            if pixelscale is None:
                self.pixelscale = self.window.shape[0] / data_shape[1]
            else:
                self.pixelscale = torch.as_tensor(
                    pixelscale, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )

    @property
    def origin(self) -> torch.Tensor:
        """
        Returns the origin (bottom-left corner) of the image window.

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the origin.
        """
        return self.window.origin

    @property
    def shape(self) -> torch.Tensor:
        """
        Returns the shape (size) of the image window.

        Returns:
                torch.Tensor: A 1D tensor of shape (2,) containing the (width, height) of the window in pixels.
        """
        return self.window.shape

    @property
    def center(self) -> torch.Tensor:
        """
        Returns the center of the image window.

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the center.
        """
        return self.window.center

    def center_alignment(self) -> torch.Tensor:
        """Determine if the center of the image is aligned at a pixel center (True)
        or if it is aligned at a pixel edge (False).

        """
        return torch.isclose(
            ((self.center - self.origin) / self.pixelscale) % 1,
            torch.tensor(0.5, dtype=AP_config.ap_dtype, device=AP_config.ap_device),
            atol=0.25,
        )

    @torch.no_grad()
    def pixel_center_alignment(self) -> torch.Tensor:
        """
        Determine the relative position of the center of a pixel with respect to the origin (mod 1)
        """
        return ((self.origin + 0.5 * self.pixelscale) / self.pixelscale) % 1

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return self.__class__(
            pixelscale=self.pixelscale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            _identity=self.identity,
            **kwargs,
        )

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        return self.__class__(
            pixelscale=self.pixelscale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window & window,
            _identity=self.identity,
            **kwargs,
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self.window.to(dtype=dtype, device=device)
        self.pixelscale.to(dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):
        if len(pixels) == 1:  # same crop in all dimension
            self.window -= pixels[0] * self.pixelscale
        elif len(pixels) == 2:  # different crop in each dimension
            self.window -= (
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                * self.pixelscale
            )
        elif len(pixels) == 4:  # different crop on all sides
            self.window -= (
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                * self.pixelscale
            )
        return self

    def get_coordinate_meshgrid_np(self, x: float = 0.0, y: float = 0.0) -> np.ndarray:
        return self.window.get_coordinate_meshgrid_np(self.pixelscale, x, y)

    def get_coordinate_meshgrid_torch(self, x=0.0, y=0.0):
        return self.window.get_coordinate_meshgrid_torch(self.pixelscale, x, y)

    def super_resolve(self, scale: int, **kwargs):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        return self.__class__(
            pixelscale=self.pixelscale / scale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            _identity=self.identity,
            **kwargs,
        )

        
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

        return self.__class__(
            pixelscale=self.pixelscale * scale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            _identity=self.identity,
            **kwargs,
        )

    def expand(self, padding: Tuple[float]) -> None:
        """
        Args:
          padding tuple[float]: length 4 tuple with amounts to pad each dimension in physical units
        """
        padding = np.array(padding)
        assert np.all(padding >= 0), "negative padding not allowed in expand method"
        pad_boundaries = tuple(np.int64(np.round(np.array(padding) / self.pixelscale)))
        self.window += tuple(padding)

    def _save_image_list(self):
        img_header = fits.Header()
        img_header["IMAGE"] = "PRIMARY"
        img_header["PXLSCALE"] = str(self.pixelscale.detach().cpu().item())
        img_header["WINDOW"] = str(self.window.get_state())
        if not self.zeropoint is None:
            img_header["ZEROPNT"] = str(self.zeropoint.detach().cpu().item())
        if not self.note is None:
            img_header["NOTE"] = str(self.note)
        return img_header

    def save(self, filename=None, overwrite=True):
        image_list = self._save_image_list()
        hdul = fits.HDUList(image_list)
        if filename is not None:
            hdul.writeto(filename, overwrite=overwrite)
        return hdul

    def load(self, filename):
        hdul = fits.open(filename)
        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PRIMARY":
                self.pixelscale = eval(hdu.header.get("PXLSCALE"))
                self.zeropoint = eval(hdu.header.get("ZEROPNT"))
                self.note = hdu.header.get("NOTE")
                self.window = Window(**eval(hdu.header.get("WINDOW")))
                break
        return hdul
    def __str__(self):
        return f"image pixelscale: {self.pixelscale} origin: {self.origin}\ndata: {self.data}"
