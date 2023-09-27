from typing import Optional, Union, Any

import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from .window_object import Window
from .. import AP_config

__all__ = ["Image_Header"]


class Image_Header:
    """Store meta-information for images to be used in AstroPhot.

    The Image_Header object stores all meta information which tells
    AstroPhot what is contained in an image array of pixels. This
    includes coordinate systems and how to transform between them (see
    :doc:`coordinates`). The image header will also know the image
    zeropoint if that data is avaialble.
    
    Args:
      window : Window or None, optional
          A Window object defining the area of the image in the coordinate
          systems. Default is None.
      filename : str or None, optional
          The name of a file containing the image data. Default is None.
      zeropoint : float or None, optional
          The image's zeropoint, used for flux calibration. Default is None.
      note : str or None, optional
          A note describing the image. Default is None.
    
    """

    north = np.pi / 2.
    
    def __init__(
        self,
        *,
        data_shape: Optional[torch.Tensor] = None,
        wcs: Optional[AstropyWCS] = None,
        window: Optional[Window] = None,
        filename: Optional[str] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        note: Optional[str] = None,
        identity: str = None,
        **kwargs: Any,
    ) -> None:
        # Record identity
        if identity is None:
            self.identity = str(id(self))
        else:
            self.identity = identity

        if filename is not None:
            self.load(filename)
            return

        # set Zeropoint
        self.zeropoint = zeropoint

        # set a note for the image
        self.note = note
        
        # Set Window
        if window is None:
            data_shape = torch.as_tensor(
                data_shape, dtype=torch.int32, device=AP_config.ap_device
            )
            # If window is not provided, create one based on provided information
            self.window = Window(
                pixel_shape=torch.flip(data_shape, (0,)),
                wcs=wcs,
                **kwargs,
            )
        else:
            # When the Window object is provided
            self.window = window
            
    @property
    def zeropoint(self):
        """The photometric zeropoint of the image, used as a flux reference
        point.

        """
        return self._zeropoint

    @zeropoint.setter
    def zeropoint(self, zp):
        if zp is None:
            self._zeropoint = None
            return

        self._zeropoint = (
            torch.as_tensor(zp, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            .clone()
            .detach()
        )
    
    @property
    def origin(self) -> torch.Tensor:
        """
        Returns the location of the origin (pixel coordinate -0.5, -0.5) of the image window in the tangent plane (arcsec).

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the origin.
        """
        return self.window.origin

    @property
    def shape(self) -> torch.Tensor:
        """
        Returns the shape (size) of the image window (arcsec, arcsec).

        Returns:
                torch.Tensor: A 1D tensor of shape (2,) containing the (width, height) of the window in arcsec.
        """
        return self.window.shape

    @property
    def center(self) -> torch.Tensor:
        """
        Returns the center of the image window (arcsec).

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the center.
        """
        return self.window.center


    def world_to_plane(self, *args, **kwargs):
        return self.window.world_to_plane(*args, **kwargs)
    def plane_to_world(self, *args, **kwargs):
        return self.window.plane_to_world(*args, **kwargs)
    def plane_to_pixel(self, *args, **kwargs):
        return self.window.plane_to_pixel(*args, **kwargs)
    def pixel_to_plane(self, *args, **kwargs):
        return self.window.pixel_to_plane(*args, **kwargs)
    def plane_to_pixel_delta(self, *args, **kwargs):
        return self.window.plane_to_pixel_delta(*args, **kwargs)
    def pixel_to_plane_delta(self, *args, **kwargs):
        return self.window.pixel_to_plane_delta(*args, **kwargs)
    def world_to_pixel(self, *args, **kwargs):
        return self.window.world_to_pixel(*args, **kwargs)
    def pixel_to_world(self, *args, **kwargs):
        return self.window.pixel_to_world(*args, **kwargs)
    def get_coordinate_meshgrid(self):
        return self.window.get_coordinate_meshgrid()
    def get_coordinate_corner_meshgrid(self):
        return self.window.get_coordinate_corner_meshgrid()
    def get_coordinate_simps_meshgrid(self):
        return self.window.get_coordinate_simps_meshgrid()

    @property
    def pixelscale(self):
        return self.window.pixelscale
    @property
    def pixel_length(self):
        return self.window.pixel_length
    @property
    def pixel_area(self):
        return self.window.pixel_area
    
    def shift(self, shift):
        """Adjust the position of the image described by the header. This will
        not adjust the data represented by the header, only the
        coordinate system that maps pixel coordinates to the plane
        coordinates.

        """
        self.window.shift(shift)

    def pixel_shift(self, shift):
        self.window.pixel_shift(shift)

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        copy_kwargs = {
            "zeropoint": self.zeropoint,
            "note": self.note,
            "window": self.window.copy(),
            "identity": self.identity,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(**copy_kwargs)

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""        
        copy_kwargs = {
            "window": self.window & window,
        }
        copy_kwargs.update(kwargs)
        return self.copy(**copy_kwargs)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        self.window.to(dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):  # fixme data_shape?
        """Reduce the size of an image by cropping some number of pixels off
        the borders. If pixels is a single value, that many pixels are
        cropped off all sides. If pixels is two values then a different
        crop is done in x vs y. If pixels is four values then crop on
        all sides are specified explicitly.

        formatted as:
        [crop all sides] or
        [crop x, crop y] or
        [crop x low, crop y low, crop x high, crop y high]

        """
        self.window.crop_pixel(pixels)
        return self

    def rescale_pixel(self, scale: int, **kwargs):
        if scale == 1:
            return self
        
        return self.copy(
            window = self.window.rescale_pixel(scale),
            **kwargs,
        )
    
    # def super_resolve(self, scale: int, **kwargs):
    #     """Increase the resolution of the referenced image by the provided
    #     scale (int).

    #     """
    #     assert isinstance(scale, int) or scale.dtype is torch.int32
    #     if scale == 1:
    #         return self

    #     return super().copy(
    #         data_shape=self.data_shape,
    #         pixelscale=self.pixelscale / scale,
    #         zeropoint=self.zeropoint,
    #         note=self.note,
    #         window=self.window.copy(),
    #         identity=self.identity,
    #         **kwargs,
    #     )

    # def reduce(self, scale: int, **kwargs):
    #     """This operation will downsample an image by the factor given. If
    #     scale = 2 then 2x2 blocks of pixels will be summed together to
    #     form individual larger pixels. A new image object will be
    #     returned with the appropriate pixelscale and data tensor. Note
    #     that the window does not change in this operation since the
    #     pixels are condensed, but the pixel size is increased
    #     correspondingly.

    #     Args:
    #         scale: factor by which to condense the image pixels. Each scale X scale region will be summed [int]

    #     """
    #     assert isinstance(scale, int) or scale.dtype is torch.int32
    #     if scale == 1:
    #         return self

    #     return super().copy(
    #         data_shape=self.data_shape,
    #         pixelscale=self.pixelscale * scale,
    #         zeropoint=self.zeropoint,
    #         note=self.note,
    #         window=self.window.copy(),
    #         identity=self.identity,
    #         **kwargs,
    #     )

    # def expand(self, padding: Tuple[float]) -> None:
    #     """
    #     Args:
    #       padding tuple[float]: length 4 tuple with amounts to pad each dimension in physical units
    #     """
    #     # fixme
    #     padding = np.array(padding)
    #     assert np.all(padding >= 0), "negative padding not allowed in expand method"
    #     pad_boundaries = tuple(np.int64(np.round(np.array(padding) / self.pixelscale)))
    #     self.window += tuple(padding)
    
    def get_state(self):
        """Returns a dictionary with necessary information to recreate the
        Image_Header object.

        """
        state = {}
        if self.zeropoint is not None:
            state["zeropoint"] = self.zeropoint.item()
        state["window"] = self.window.get_state()
        if self.note is not None:
            state["note"] = self.note
        return state

    def _save_image_list(self):
        """
        Constructs a FITS header object which has the necessary information to recreate the Image_Header object.
        """
        img_header = fits.Header()
        img_header["IMAGE"] = "PRIMARY"
        img_header["WINDOW"] = str(self.window.get_state())
        if not self.zeropoint is None:
            img_header["ZEROPNT"] = str(self.zeropoint.detach().cpu().item())
        if not self.note is None:
            img_header["NOTE"] = str(self.note)
        return img_header

    def set_fits_state(self, state):
        """
        Updates the state of the Image_Header using information saved in a FITS header (more generally, a properly formatted dictionary will also work).
        """
        self.zeropoint = eval(state.get("ZEROPNT", "None"))
        self.note = state.get("NOTE", None)
        self.window = Window(state=eval(state["WINDOW"]))
        
    def save(self, filename=None, overwrite=True):
        """
        Save header to a FITS file.
        """
        image_list = self._save_image_list()
        hdul = fits.HDUList(image_list)
        if filename is not None:
            hdul.writeto(filename, overwrite=overwrite)
        return hdul

    def load(self, filename):
        """
        load header from a FITS file.
        """
        hdul = fits.open(filename)
        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PRIMARY":
                self.set_fits_state(hdu.header)
                break
        return hdul

    def __str__(self):
        state = self.get_state()
        state.update(self.window.get_state())
        keys = ["pixel_shape", "pixelscale", "reference_imageij", "reference_imagexy"]
        if "zeropoint" in state:
            keys.append("zeropoint")
        if "note" in state:
            keys.append("note")
        return "\n".join(f"{key}: {state[key]}" for key in keys)

    def __repr__(self):
        state = self.get_state()
        state.update(self.window.get_state())
        return "\n".join(f"{key}: {state[key]}" for key in sorted(state.keys()))
