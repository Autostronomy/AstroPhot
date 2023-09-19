from typing import Optional, Union, Any, Sequence, Tuple
from copy import deepcopy

import torch
from torch.nn.functional import pad
import numpy as np
from astropy.io import fits

from .window_object import Window, Window_List
from .wcs import WPCS
from ..utils.conversions.units import deg_to_arcsec
from .. import AP_config

__all__ = ["Image_Header"]


class Image_Header(WPCS):
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
        The origin of the image in the tangent plane coordinate system, as a 1D array of length 2. Default is None.
    origin_radec : numpy.ndarray or None, optional
        The origin of the image in the world coordinate system (RA, DEC), as a 1D array of length 2. Default is None.
    center : numpy.ndarray or None, optional
        The center of the image in the tangent plane coordinate system, as a 1D array of length 2. Default is None.
    center_radec : numpy.ndarray or None, optional
        The center of the image in the world coordinate system (RA, DEC), as a 1D array of length 2. Default is None.

    Returns:
    --------
    None
    """

    north = np.pi / 2.
    default_reference_imageij = (-0.5,-0.5)
    default_reference_imagexy = (0,0)
    default_pixelscale = 1
    
    def __init__(
        self,
        data_shape: Optional[torch.Tensor],
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        wcs: Optional["astropy.wcs.wcs.WCS"] = None,
        window: Optional[Window] = None,
        filename: Optional[str] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        note: Optional[str] = None,
        origin: Optional[Sequence] = None,
        origin_radec: Optional[Sequence] = None,
        center: Optional[Sequence] = None,
        center_radec: Optional[Sequence] = None,
        reference_imagexy: Optional[Sequence] = None,
        reference_imageij: Optional[Sequence] = None,
        wpcs: Optional["WPCS"] = None,
        ppcs: Optional["PPCS"] = None,
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

        self.data_shape = torch.as_tensor(
            data_shape, dtype=torch.int32, device=AP_config.ap_device
        )
        # set Zeropoint
        self.zeropoint = zeropoint

        # set a note for the image
        self.note = note

        if wcs is not None and pixelscale is None:
            self.pixelscale = deg_to_arcsec * wcs.pixel_scale_matrix
        else:
            if wcs is not None and isinstance(pixelscale, float):
                AP_config.ap_logger.warn(
                    "Overriding WCS pixelscale with manual input! To remove this message, either let WCS define pixelscale, or input full pixelscale matrix"
                )
            self.pixelscale = pixelscale

        # Set Window
        if window is None:
            # If window is not provided, create one based on pixelscale and data shape
            assert (
                self.pixelscale is not None
            ), "pixelscale cannot be None if window is not provided"

            end = self.pixel_to_plane_delta(
                torch.flip(
                    torch.tensor(
                        data_shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                    ),
                    (0,),
                )
            )
            shape = torch.stack(
                (
                    torch.linalg.norm(
                        self.pixel_to_plane_delta(
                            torch.tensor(
                                [data_shape[1], 0],
                                dtype=AP_config.ap_dtype,
                                device=AP_config.ap_device,
                            )
                        )
                    ),
                    torch.linalg.norm(
                        self.pixel_to_plane_delta(
                            torch.tensor(
                                [0, data_shape[0]],
                                dtype=AP_config.ap_dtype,
                                device=AP_config.ap_device,
                            )
                        )
                    ),
                )
            )
            if wcs is not None:  # Image coordinates provided by WCS
                self.reference_imageij = wcs.wcs.crpix if reference_imageij is None else reference_imageij
                self.reference_imagexy = (0,0) if reference_imagexy is None else reference_imagexy
                kwargs["reference_radec"] = kwargs.get("reference_radec", wcs.wcs.crval)
                super().__init__(**kwargs)
                origin = torch.stack(self.pixel_to_plane(*(-0.5 * torch.ones_like(self.reference_imageij))))
            elif (
                origin_radec is not None
            ):  # Image reference position from RA and DEC of image origin
                # Origin given, it is reference point
                self.reference_imageij = (-0.5,-0.5)
                origin_radec = torch.as_tensor(
                    origin_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                kwargs["reference_radec"] = kwargs.get("reference_radec", origin_radec)
                super().__init__(**kwargs)
                origin = torch.stack(self.world_to_plane(*origin_radec))
                self.reference_imagexy = origin
            elif (
                center_radec is not None
            ):  # Image reference position from RA and DEC of image center
                pix_center = torch.flip(
                    data_shape.to(dtype=AP_config.ap_dtype),
                    (0,),
                ) / 2 - 0.5

                self.reference_imageij = pix_center
                center_radec = torch.as_tensor(
                    center_radec, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                kwargs["reference_radec"] = kwargs.get("reference_radec", center_radec)
                super().__init__(**kwargs)
                center = torch.stack(self.world_to_plane(*center_radec))
                self.reference_imagexy = center
                origin = self.pixel_to_plane(*(-0.5 * torch.ones_like(self.reference_imagexy)))
            elif (
                origin is not None
            ):  # Image reference position from tangent plane position of image origin
                self.reference_imageij = (-0.5,-0.5)
                self.reference_imagexy = origin
                super().__init__(**kwargs)
            elif (
                center is not None
            ):  # Image reference position from tangent plane position of image center
                pix_center = torch.flip(
                    data_shape.to(dtype=AP_config.ap_dtype),
                    (0,),
                ) / 2 - 0.5
                self.reference_imageij = pix_center
                self.reference_imagexy = center
                super().__init__(**kwargs)
                origin = self.pixel_to_plane(*(-0.5 * torch.ones_like(self.reference_imagexy)))                
            else:  # Image origin assumed to be at tangent plane origin
                super().__init__(**kwargs)
                self.reference_imageij = (-0.5,-0.5) if reference_imageij is None else reference_imageij
                self.reference_imagexy = (0,0) if reference_imagexy is None else reference_imagexy
                origin = self.pixel_to_plane(*(-0.5 * torch.ones_like(self.reference_imagexy)))

            self.window = Window(
                origin=origin,
                shape=shape,
                pixelshape=self.pixelscale,
                projection=self.projection,
                reference_radec=self.reference_radec,
                reference_planexy=self.reference_planexy,
            )
        else:
            # When the Window object is provided
            self.window = window
            kwargs["reference_radec"] = window.reference_radec
            kwargs["reference_planexy"] = window.reference_planexy
            kwargs["projection"] = window.projection
            super().__init__(**kwargs)
            self.reference_imageij = (-0.5,-0.5) if reference_imageij is None else reference_imageij
            self.reference_imagexy = (0,0) if reference_imagexy is None else reference_imagexy
            if self.pixelscale is None:
                pixelscale = self.window.shape[0] / data_shape[1]
                AP_config.ap_logger.warn(
                    "Assuming square pixels with pixelscale f{pixelscale.item()}. To remove this warning please provide the pixelscale explicitly when creating an image."
                )
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
        self._pixelscale_inv = torch.linalg.inv(self.pixelscale)

    @property
    def pixel_area(self):
        return self._pixel_area

    @property
    def pixel_length(self):
        return self._pixel_length

    @property
    def reference_imageij(self):
        """pixel coordiantes where the pixel grid is fixed to the tangent
        plane. These should be in pixel units where (0,0) is the
        center of the [0,0] indexed pixel. However, it is still in xy
        format, meaning that the first index gives translations in the
        x-axis (horizontal-axis) of the image.

        """
        return self._reference_imageij

    @reference_imageij.setter
    def reference_imageij(self, imageij):
        self._reference_imageij = torch.as_tensor(
            imageij, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
    @property
    def reference_imagexy(self):
        """plane coordiantes where the image grid is fixed to the tangent
        plane. These should be in arcsec.

        """
        return self._reference_imagexy

    @reference_imagexy.setter
    def reference_imagexy(self, imagexy):
        self._reference_imagexy = torch.as_tensor(
            imagexy, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def pixel_to_plane(self, pixel_i, pixel_j=None):
        """Take in a coordinate on the regular pixel grid, where
        0,0 is the center of the [0,0] indexed pixel. This coordinate is
        transformed into the tangent plane coordiante system based on the
        pixel scale and fixme
        """
        if pixel_j is None:
            return torch.stack(self.pixel_to_plane(*pixel_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_i.reshape(-1), pixel_j.reshape(-1))) - self.reference_imageij.view(2,1)) + self.reference_imagexy.view(2,1)
        return coords[0].reshape(pixel_i.shape), coords[1].reshape(pixel_j.shape)

    def plane_to_pixel(self, plane_x, plane_y=None):
        if plane_y is None:
            return torch.stack(self.plane_to_pixel(*plane_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_x.reshape(-1), plane_y.reshape(-1))) - self.reference_imagexy.view(2,1)) + self.reference_imageij.view(2,1)
        return coords[0].reshape(plane_x.shape), coords[1].reshape(plane_y.shape)

    def pixel_to_plane_delta(self, pixel_delta_i, pixel_delta_j=None):
        """Take in a coordinate on the regular cartesian pixel grid, where
        0,0 is the center of the first pixel. This coordinate is
        transformed into the plane coordiante system based on the
        pixel scale and origin position for this image. In the plane
        coordinate system the origin is placed with respect to the
        bottom corner of the 0,0 pixel.

        """
        if pixel_delta_j is None:
            return torch.stack(self.pixel_to_plane_delta(*pixel_delta_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_delta_i.reshape(-1), pixel_delta_j.reshape(-1))))
        return coords[0].reshape(pixel_delta_i.shape), coords[1].reshape(pixel_delta_j.shape)

    def plane_to_pixel_delta(self, plane_delta_x, plane_delta_y=None):
        if plane_delta_y is None:
            return torch.stack(self.plane_to_pixel_delta(*plane_delta_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_delta_x.reshape(-1), plane_delta_y.reshape(-1))))
        return coords[0].reshape(plane_delta_x.shape), coords[1].reshape(plane_delta_y.shape)

    def world_to_pixel(self, world_RA, world_DEC):
        return self.plane_to_pixel(*self.world_to_plane(world_DEC, world_RA))
    def pixel_to_world(self, pixel_i, pixel_j):
        return self.plane_to_world(*self.pixel_to_plane(pixel_i, pixel_j))
    
    @property
    def zeropoint(self):
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
        coordinate system that maps pixel coordinates to the plane
        coordinates.

        """
        self.window.shift_origin(shift)
        self.reference_imagexy = self.reference_imagexy + shift

    def pixel_shift_origin(self, shift):
        self.shift_origin(self.pixel_to_plane_delta(shift))

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return super().copy(
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
        return super().copy(
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
        super().to(dtype=dtype, device=device)
        self.window.to(dtype=dtype, device=device)
        self.pixelscale.to(dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):  # fixme data_shape
        if len(pixels) == 1:  # same crop in all dimension
            self.window -= self.pixel_to_plane_delta(
                torch.as_tensor(
                    [pixels[0], pixels[0]],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
            ).abs()
        elif len(pixels) == 2:  # different crop in each dimension
            self.window -= self.pixel_to_plane_delta(
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            ).abs()
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            low = self.pixel_to_plane_delta(pixels[:2])
            high = self.pixel_to_plane_delta(pixels[2:])
            self.window -= torch.cat((low, high)).abs()
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
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

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
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

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
        Coords = self.pixel_to_plane(meshx, meshy)
        return torch.stack(Coords)

    def super_resolve(self, scale: int, **kwargs):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        return super().copy(
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

        return super().copy(
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
        state = super().get_state()
        state["pixelscale"] = self.pixelscale.tolist()
        if self.zeropoint is not None:
            state["zeropoint"] = self.zeropoint.item()
        state["window"] = self.window.get_state()
        if self.note is not None:
            state["note"] = self.note
        return state

    def _save_image_list(self):
        img_header = fits.Header(super().get_fits_state())
        img_header["IMAGE"] = "PRIMARY"
        img_header["PXLSCALE"] = str(self.pixelscale.detach().cpu().tolist())
        img_header["WINDOW"] = str(self.window.get_state())
        if not self.zeropoint is None:
            img_header["ZEROPNT"] = str(self.zeropoint.detach().cpu().item())
        if not self.note is None:
            img_header["NOTE"] = str(self.note)
        return img_header

    def set_fits_state(self, state):
        super().set_fits_state(state)
        self.pixelscale = eval(hdu.header.get("PXLSCALE"))
        self.zeropoint = eval(hdu.header.get("ZEROPNT"))
        self.note = hdu.header.get("NOTE")
        self.window = Window(state=eval(hdu.header.get("WINDOW")))
        
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
                self.set_fits_state(hdu.header)
                break
        return hdul

    def __str__(self):
        state = self.get_state()
        return "\n".join(f"{key}: {state[key]}" for key in state)
