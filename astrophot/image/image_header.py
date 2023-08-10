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
    north = np.pi / 2.0

    def __init__(
        self,
        data_shape: Optional[torch.Tensor] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        wcs: Optional["astropy.wcs.wcs.WCS"] = None,
        window: Optional[Window] = None,
        filename: Optional[str] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        note: Optional[str] = None,
        origin: Optional[Sequence] = None,
        center: Optional[Sequence] = None,
        identity: str = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of the APImage class.

        Parameters:
        -----------
        pixelscale : float or None, optional
            The physical scale of the pixels in the image, this is represented as a matrix which projects pixel units into sky units: $pixelscale @ pixel_vec = sky_vec$. The pixel scale matrix can be thought of in four components: \vec{s} @ F @ R @ S where \vec{s} is the side length of the pixels, F is a diagonal matrix of {1,-1} which flips the axes orientation, R is a rotation matrix, and S is a shear matrix which turns rectangular pixels into parallelograms. Default is None.
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
        if identity is None:
            self.identity = str(id(self))
        else:
            self.identity = identity

        if filename is not None:
            self.load(filename)
            return

        self.data_shape = torch.as_tensor(
            data_shape, dtype=torch.int32, device=AP_config.ap_device
        )
        # set Zeropoint
        if zeropoint is None:
            self.zeropoint = None
        else:
            self.zeropoint = torch.as_tensor(
                zeropoint, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

        # set a note for the image
        self.note = note

        if wcs is not None and pixelscale is None:
            self.pixelscale = wcs.pixel_scale_matrix
        else:
            self.pixelscale = pixelscale

        # Set Window
        if window is None:
            # If window is not provided, create one based on pixelscale and data shape
            assert (
                self.pixelscale is not None
            ), "pixelscale cannot be None if window is not provided"

            end = self.pixel_to_world_delta(
                torch.flip(
                    torch.tensor(
                        data_shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                    ),
                    (0,),
                )
            )
            shape = torch.linalg.solve(self.pixelscale / self.pixel_length, end)
            if wcs is not None:
                wcs_origin = wcs.pixel_to_world(-0.5, -0.5)
                origin = torch.as_tensor(
                    [wcs_origin.ra.arcsec, wcs_origin.dec.arcsec],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
            elif origin is None and center is None:
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
                    - end / 2
                )

            self.window = Window(origin=origin, shape=shape, projection=self.pixelscale)
        else:
            # When The Window object is provided
            self.window = window
            if self.pixelscale is None:
                pixelscale = self.window.shape[0] / data_shape[1]
                self.pixelscale = torch.tensor(
                    [[pixelscale, 0.0], [0.0, pixelscale]],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )

    @property
    def pixelscale(self):
        return self._pixelscale

    @pixelscale.setter
    def pixelscale(self, pix):
        if pix is None:
            self._pixelscale = None
            return

        self._pixelscale = (
            torch.as_tensor(pix, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            .clone()
            .detach()
        )
        if self._pixelscale.numel() == 1:
            self._pixelscale = torch.tensor(
                [[self._pixelscale.item(), 0.0], [0.0, self._pixelscale.item()]],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        self._pixel_area = torch.linalg.det(self.pixelscale).abs()
        self._pixel_length = self._pixel_area.sqrt()
        self._pixel_origin = None
        self._pixelscale_inv = torch.linalg.inv(self.pixelscale)

    @property
    def pixel_area(self):
        return self._pixel_area

    @property
    def pixel_length(self):
        return self._pixel_length

    @property
    def pixel_origin(self):
        if self._pixel_origin is None:
            self._pixel_origin = self.origin + self.pixel_to_world_delta(
                0.5 * torch.ones_like(self.origin)
            )
        return self._pixel_origin

    def pixel_to_world(self, pixel_coordinate, internal_transpose=False):
        """Take in a coordinate on the regular cartesian pixel grid, where
        0,0 is the center of the first pixel. This coordinate is
        transformed into the world coordiante system based on the
        pixel scale and origin position for this image. In the world
        coordinate system the origin is placed with respect to the
        bottom corner of the 0,0 pixel.

        """
        if internal_transpose:
            return (self.pixelscale @ pixel_coordinate).T + self.pixel_origin
        return (self.pixelscale @ pixel_coordinate) + self.pixel_origin

    def world_to_pixel(self, world_coordinate, unsqueeze_origin=False):
        if unsqueeze_origin:
            O = self.pixel_origin.unsqueeze(-1)
        else:
            O = self.pixel_origin
        return self._pixelscale_inv @ (world_coordinate - O)

    def pixel_to_world_delta(self, pixel_delta):
        """Take in a coordinate on the regular cartesian pixel grid, where
        0,0 is the center of the first pixel. This coordinate is
        transformed into the world coordiante system based on the
        pixel scale and origin position for this image. In the world
        coordinate system the origin is placed with respect to the
        bottom corner of the 0,0 pixel.

        """
        return self.pixelscale @ pixel_delta

    def world_to_pixel_delta(self, world_delta):
        return self._pixelscale_inv @ world_delta

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

    def shift_origin(self, shift):
        """Adjust the origin position of the image header. This will not
        adjust the data represented by the header, only the
        coordiantes system that maps pixel coordinates to the world
        coordinates.

        """
        self.window.shift_origin(shift)
        self._pixel_origin = None

    def pixel_shift_origin(self, shift):
        self.shift_origin(self.pixel_to_world_delta(shift))

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return self.__class__(
            data_shape=self.data_shape,
            pixelscale=self.pixelscale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            identity=self.identity,
            **kwargs,
        )

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        indices = window.get_indices(self)
        return self.__class__(
            data_shape=(
                indices[0].stop - indices[0].start,
                indices[1].stop - indices[1].start,
            ),
            pixelscale=self.pixelscale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window & window,
            identity=self.identity,
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

    def crop(self, pixels):  # fixme data_shape
        if len(pixels) == 1:  # same crop in all dimension
            self.window -= self.pixel_to_world_delta(
                torch.as_tensor(
                    [pixels[0], pixels[0]],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
            )
        elif len(pixels) == 2:  # different crop in each dimension
            self.window -= self.pixel_to_world_delta(
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            )
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            low = self.pixel_to_world_delta(pixels[:2])
            high = self.pixel_to_world_delta(pixels[2:])
            self.window -= torch.cat((low, high))
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        self._pixel_origin = None
        return self

    @torch.no_grad()
    def get_coordinate_meshgrid(self):
        n_pixels = self.data_shape
        xsteps = torch.arange(
            n_pixels[1], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        ysteps = torch.arange(
            n_pixels[0], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_world(
            torch.stack((meshx, meshy)).view(2, -1), internal_transpose=True
        ).T
        return Coords.reshape((2, *meshx.shape))

    @torch.no_grad()
    def get_coordinate_corner_meshgrid(self):
        n_pixels = self.data_shape
        xsteps = (
            torch.arange(
                n_pixels[1] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            - 0.5
        )
        ysteps = (
            torch.arange(
                n_pixels[0] + 1, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            - 0.5
        )
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_world(
            torch.stack((meshx, meshy)).view(2, -1), internal_transpose=True
        ).T
        return Coords.reshape((2, *meshx.shape))

    @torch.no_grad()
    def get_coordinate_simps_meshgrid(self):
        n_pixels = self.data_shape
        xsteps = (
            0.5
            * torch.arange(
                2 * (n_pixels[1]) + 1,
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - 0.5
        )
        ysteps = (
            0.5
            * torch.arange(
                2 * (n_pixels[0]) + 1,
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            - 0.5
        )
        meshx, meshy = torch.meshgrid(
            xsteps,
            ysteps,
            indexing="xy",
        )
        Coords = self.pixel_to_world(
            torch.stack((meshx, meshy)).view(2, -1), internal_transpose=True
        ).T
        return Coords.reshape((2, *meshx.shape))

    def super_resolve(self, scale: int, **kwargs):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        return self.__class__(
            data_shape=self.data_shape,
            pixelscale=self.pixelscale / scale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            identity=self.identity,
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
            data_shape=self.data_shape,
            pixelscale=self.pixelscale * scale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.copy(),
            identity=self.identity,
            **kwargs,
        )

    def expand(self, padding: Tuple[float]) -> None:
        """
        Args:
          padding tuple[float]: length 4 tuple with amounts to pad each dimension in physical units
        """
        # fixme
        padding = np.array(padding)
        assert np.all(padding >= 0), "negative padding not allowed in expand method"
        pad_boundaries = tuple(np.int64(np.round(np.array(padding) / self.pixelscale)))
        self.window += tuple(padding)

    def get_state(self):
        state = {}
        state["pixelscale"] = self.pixelscale.tolist()
        if self.zeropoint is not None:
            state["zeropoint"] = self.zeropoint.item()
        state["window"] = self.window.get_state()
        if self.note is not None:
            state["note"] = self.note
        return state

    def _save_image_list(self):
        img_header = fits.Header()
        img_header["IMAGE"] = "PRIMARY"
        img_header["PXLSCALE"] = str(self.pixelscale.detach().cpu().tolist())
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
                self.pixelscale = torch.tensor(
                    eval(hdu.header.get("PXLSCALE")),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                self.zeropoint = torch.tensor(
                    eval(hdu.header.get("ZEROPNT")),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                self.note = hdu.header.get("NOTE")
                self.window = Window(state=eval(hdu.header.get("WINDOW")))
                break
        return hdul

    def __str__(self):
        state = self.get_state()
        return "\n".join(f"{key}: {state[key]}" for key in state)
