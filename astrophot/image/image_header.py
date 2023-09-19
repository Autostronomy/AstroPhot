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
    """Store meta-information for images to be used in AstroPhot.

    The Image_Header object stores all meta information which tells
    AstroPhot what is contained in an image array of pixels. This
    includes information about where the pixels are in the coordiante
    systems and how to transform between them (see
    :doc:`coordiantes`). The image header will also know the image
    zeropoint if that data is avaialble.

    There are several ways to tell an Image_Header object where to
    place the pixels it stores. The simplest method is to pass an
    Astropy WCS object such as::

      H = ap.image.Image_Header(
          data_shape = data.shape,
          wcs = wcs,
      )

    this will automatically place your image at the correct RA, DEC
    and assign the correct pixel scale. WARNING, it will default to
    setting the reference RA DEC at the reference RA DEC of the wcs
    object; if you have multiple images you should force them all to
    have the same reference world coordiante by passing
    ``reference_radec = (ra, dec)``. See the :doc:`coordinates`
    documentation for more details. There are several other ways to
    initialize the image header. If you provide ``origin_radec`` then
    it will place the image origin at the requested RA DEC
    coordinates. If you provide ``center_radec`` then it will place
    the image center at the requested RA DEC coordiantes. Note that in
    these cases the fixed point between the pixel grid and image plane
    is different (pixel origin and center respectively); so if you
    have rotated pixels in your pixel scale matrix then everything
    will be rotated about different points (pixel origin and center
    respectively). If you provide ``origin`` or ``center`` then those
    are coordiantes in the tangent plane (arcsec) and they will
    correspondingly become fixed points. For arbitrary control over
    the pixel positioning, use ``reference_imageij`` and
    ``reference_imagexy`` to fix the pixel and tangent plane
    coordinates respectively to each other, any rotation or shear will
    happen about that fixed point.
    
    Args:
    pixelscale : float or None, optional
        The physical scale of the pixels in the image, this is
        represented as a matrix which projects pixel units into sky
        units: :math:`pixelscale @ pixel_vec = sky_vec`. The pixel
        scale matrix can be thought of in four components:
        :math:`\vec{s} @ F @ R @ S` where :math:`\vec{s}` is the side
        length of the pixels, :math:`F` is a diagonal matrix of {1,-1}
        which flips the axes orientation, :math:`R` is a rotation
        matrix, and :math:`S` is a shear matrix which turns
        rectangular pixels into parallelograms. Default is None.
    window : Window or None, optional
        A Window object defining the area of the image on the tangent
        plane to use. Default is None.
    filename : str or None, optional
        The name of a file containing the image data. Default is None.
    zeropoint : float or None, optional
        The image's zeropoint, used for flux calibration. Default is None.
    note : str or None, optional
        A note describing the image. Default is None.
    origin : Sequence or None, optional
        The origin of the image in the tangent plane coordinate system
        (arcsec), as a 1D array of length 2. Default is None.
    origin_radec : Sequence or None, optional
        The origin of the image in the world coordinate system (RA,
        DEC in degrees), as a 1D array of length 2. Default is None.
    center : Sequence or None, optional
        The center of the image in the tangent plane coordinate system
        (arcsec), as a 1D array of length 2. Default is None.
    center_radec : Sequence or None, optional
        The center of the image in the world coordinate system (RA,
        DEC in degrees), as a 1D array of length 2. Default is None.
    reference_imageij : Sequence or None, optional
        The pixel coordinate at which the image is fixed to the
        tangent plane. By default this is (-0.5, -0.5) or the bottom
        corner of the [0,0] indexed pixel.
    reference_imagexy : Sequence or None, optional
        The tangent plane coordinate at which the image is fixed,
        corresponding to the reference_imageij coordinate. These two
        reference points ar pinned together, any rotations would occur
        about this point. By default this is (0., 0.).
    
    """

    north = np.pi / 2.
    default_reference_imageij = (-0.5,-0.5)
    default_reference_imagexy = (0,0)
    default_pixelscale = 1
    
    def __init__(
        self,
        data_shape: Optional[torch.Tensor],
        *,
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

        if wcs is not None:
            if wcs.wcs.ctype[0] != "RA---TAN":
                AP_config.ap_logger.warn("Astropy WCS not tangent plane coordinate system!")
            if wcs.wcs.ctype[1] != "DEC--TAN":
                AP_config.ap_logger.warn("Astropy WCS not tangent plane coordinate system!")
                
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
            assert sum(C is not None for C in [wcs, origin_radec, center_radec, origin, center]) <= 1, "Please provide only one reference position for the image, otherwise the placement is ambiguous"
            if wcs is not None:  # Image coordinates provided by WCS
                kwargs["reference_radec"] = kwargs.get("reference_radec", wcs.wcs.crval)
                super().__init__(**kwargs)
                self.reference_imageij = wcs.wcs.crpix
                self.reference_imagexy = torch.stack(self.world_to_plane(*torch.tensor(wcs.wcs.crval, dtype=AP_config.ap_dtype, device=AP_config.ap_device)))
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
                    self.data_shape.to(dtype=AP_config.ap_dtype),
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
                    self.data_shape.to(dtype=AP_config.ap_dtype),
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
            self.reference_imagexy = window.origin if reference_imagexy is None else reference_imagexy
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
        """Matrix defining the shape of pixels in the tangent plane, these
        can be any parallelogram defined by the matrix.

        """
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
        """The area inside a pixel in arcsec^2

        """
        return self._pixel_area

    @property
    def pixel_length(self):
        """The approximate length of a pixel, which is just
        sqrt(pixel_area). For square pixels this is the actual pixel
        length, for rectangular pixels it is a kind of average.

        The pixel_length is typically not used for exact calculations
        and instead sets a size scale within an image.

        """
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
        """Take in a coordinate on the regular pixel grid, where 0,0 is the
        center of the [0,0] indexed pixel. This coordinate is
        transformed into the tangent plane coordiante system (arcsec)
        based on the pixel scale and reference positions. If the pixel
        scale matrix is :math:`P`, the reference pixel is
        :math:`\vec{r}_{pix}`, the reference tangent plane point is
        :math:`\vec{r}_{tan}`, and the coordinate to transform is
        :math:`\vec{c}_{pix}` then the coordiante in the tangent plane
        is:

        .. math::
            \vec{c}_{tan} = [P(\vec{c}_{pix} - \vec{r}_{pix})] + \vec{r}_{tan}

        """
        if pixel_j is None:
            return torch.stack(self.pixel_to_plane(*pixel_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_i.reshape(-1), pixel_j.reshape(-1))) - self.reference_imageij.view(2,1)) + self.reference_imagexy.view(2,1)
        return coords[0].reshape(pixel_i.shape), coords[1].reshape(pixel_j.shape)

    def plane_to_pixel(self, plane_x, plane_y=None):
        """Take a coordinate on the tangent plane (arcsec) and transform it to
        the cooresponding pixel grid coordinate (pixel units where
        (0,0) is the [0,0] indexed pixel). Transformation is done
        based on the pixel scale and reference positions. If the pixel
        scale matrix is :math:`P`, the reference pixel is
        :math:`\vec{r}_{pix}`, the reference tangent plane point is
        :math:`\vec{r}_{tan}`, and the coordinate to transform is
        :math:`\vec{c}_{tan}` then the coordiante in the pixel grid
        is:

        .. math::
            \vec{c}_{pix} = [P^{-1}(\vec{c}_{tan} - \vec{r}_{tan})] + \vec{r}_{pix}

        """
        if plane_y is None:
            return torch.stack(self.plane_to_pixel(*plane_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_x.reshape(-1), plane_y.reshape(-1))) - self.reference_imagexy.view(2,1)) + self.reference_imageij.view(2,1)
        return coords[0].reshape(plane_x.shape), coords[1].reshape(plane_y.shape)

    def pixel_to_plane_delta(self, pixel_delta_i, pixel_delta_j=None):
        """Take a translation in pixel space and determine the cooresponding
        translation in the tangent plane (arcsec). Essentially this performs
        the pixel scale matrix multiplication without any reference
        coordinates applied.

        """
        if pixel_delta_j is None:
            return torch.stack(self.pixel_to_plane_delta(*pixel_delta_i))
        coords = torch.mm(self.pixelscale, torch.stack((pixel_delta_i.reshape(-1), pixel_delta_j.reshape(-1))))
        return coords[0].reshape(pixel_delta_i.shape), coords[1].reshape(pixel_delta_j.shape)

    def plane_to_pixel_delta(self, plane_delta_x, plane_delta_y=None):
        """Take a translation in tangent plane space (arcsec) and determine
        the cooresponding translation in pixel space. Essentially this
        performs the pixel scale matrix multiplication without any
        reference coordinates applied.

        """
        if plane_delta_y is None:
            return torch.stack(self.plane_to_pixel_delta(*plane_delta_x))
        coords = torch.mm(self._pixelscale_inv, torch.stack((plane_delta_x.reshape(-1), plane_delta_y.reshape(-1))))
        return coords[0].reshape(plane_delta_x.shape), coords[1].reshape(plane_delta_y.shape)

    def world_to_pixel(self, world_RA, world_DEC=None):
        """A wrapper which applies :meth:`world_to_plane` then
        :meth:`plane_to_pixel`, see those methods for further
        information.

        """
        if world_DEC is None:
            return torch.stack(self.world_to_pixel(*world_RA))
        return self.plane_to_pixel(*self.world_to_plane(world_RA, world_DEC))
    def pixel_to_world(self, pixel_i, pixel_j=None):
        """A wrapper which applies :meth:`pixel_to_plane` then
        :meth:`plane_to_world`, see those methods for further
        information.

        """
        if pixel_j is None:
            return torch.stack(self.pixel_to_world(*pixel_i))
        return self.plane_to_world(*self.pixel_to_plane(pixel_i, pixel_j))
    
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
        Returns the origin (pixel coordinate -0.5, -0.5) of the image window in the tangent plane (arcsec).

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the origin.
        """
        return self.window.origin

    @property
    def shape(self) -> torch.Tensor:
        """
        Returns the shape (size) of the image window in arcsec.

        Returns:
                torch.Tensor: A 1D tensor of shape (2,) containing the (width, height) of the window in pixels.
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

    def shift_origin(self, shift):
        """Adjust the position of the image described by the header. This will
        not adjust the data represented by the header, only the
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
            window=self.window & window,#fixme, need reference_imagexy and reference_imageij defined to remain the same
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
        self._reference_imageij = self._reference_imageij.to(dtype=dtype, device=device)
        self._reference_imagexy = self._reference_imagexy.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):  # fixme data_shape?
        """Reduce the size of an image by cropping some number of pixels off
        the borders. If pixels is a single value, that many pixels are
        cropped off all sides. If pixels is two values thena different
        crop is done in x vs y. If pixels is four values then crop on
        all sides are specified explicitly.

        """
        if len(pixels) == 1:  # same crop in all dimension
            pix_shift = torch.as_tensor(
                [pixels[0], pixels[0]],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            self.window -= self.pixel_to_plane_delta(
                pix_shift
            ).abs()
            self.reference_imageij = self.reference_imageij - pix_shift
            self.data_shape = self.data_shape - 2 * pix_shift
        elif len(pixels) == 2:  # different crop in each dimension
            pix_shift = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.window -= self.pixel_to_plane_delta(
                pix_shift
            ).abs()
            self.reference_imageij = self.reference_imageij - pix_shift
            self.data_shape = self.data_shape - 2 * pix_shift
        elif len(pixels) == 4:  # different crop on all sides
            pixels = torch.as_tensor(
                pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            low = self.pixel_to_plane_delta(pixels[:2])
            high = self.pixel_to_plane_delta(pixels[2:])
            self.window -= torch.cat((low, high)).abs()
            self.reference_imageij = self.reference_imageij - low
            self.data_shape = self.data_shape - low - high
        else:
            raise ValueError(f"Unrecognized pixel crop format: {pixels}")
        return self

    @torch.no_grad()
    def get_coordinate_meshgrid(self):
        """Returns a meshgrid with tangent plane coordinates for the center
        of every pixel.

        """
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
        """Returns a meshgrid with tangent plane coordinates for the corners
        of every pixel.

        """
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
        """Returns a meshgrid with tangent plane coordinates for performing
        simpsons method pixel integration (all corners, centers, and
        middle of each edge). This is approximately 4 times more
        points than the standard :meth:`get_coordinate_meshgrid`.

        """
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
        """Increase the resolution of the referenced image by the provided
        scale (int).

        """
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

        Args:
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
        """Returns a dictionary with necessary information to recreate the
        Image_Header object.

        """
        state = super().get_state()
        state["data_shape"] = self.data_shape.detach().cpu().tolist()
        state["pixelscale"] = self.pixelscale.detach().cpu().tolist()
        state["reference_imageij"] = self.reference_imageij.detach().cpu().tolist()
        state["reference_imagexy"] = self.reference_imagexy.detach().cpu().tolist()
        if self.zeropoint is not None:
            state["zeropoint"] = self.zeropoint.item()
        state["window"] = self.window.get_state()
        if self.note is not None:
            state["note"] = self.note
        return state

    def _save_image_list(self):
        """
        Constructs a fits header object which has the necessary information to recreate the Image_Header object.
        """
        img_header = fits.Header(super().get_fits_state())
        img_header["IMAGE"] = "PRIMARY"
        img_header["PXLSCALE"] = str(self.pixelscale.detach().cpu().tolist())
        img_header["WINDOW"] = str(self.window.get_state())
        img_header["REFIMIJ"] = str(self.reference_imageij.detach().cpu().tolist())
        img_header["REFIMXY"] = str(self.reference_imagexy.detach().cpu().tolist())
        if not self.zeropoint is None:
            img_header["ZEROPNT"] = str(self.zeropoint.detach().cpu().item())
        if not self.note is None:
            img_header["NOTE"] = str(self.note)
        return img_header

    def set_fits_state(self, state):
        """
        Updates the state of the Image_Header using information saved in a fits header.
        """
        super().set_fits_state(state)
        self.pixelscale = eval(state["PXLSCALE"])
        self.zeropoint = eval(state.get("ZEROPNT", "None"))
        self.reference_imageij = eval(state["REFIMIJ"])
        self.reference_imagexy = eval(state["REFIMXY"])
        self.note = state.get("NOTE",None)
        self.window = Window(state=eval(state["WINDOW"]))
        
    def save(self, filename=None, overwrite=True):
        """
        Save to a fits file.
        """
        image_list = self._save_image_list()
        hdul = fits.HDUList(image_list)
        if filename is not None:
            hdul.writeto(filename, overwrite=overwrite)
        return hdul

    def load(self, filename):
        """
        load from a fits file.
        """
        hdul = fits.open(filename)
        for hdu in hdul:
            if "IMAGE" in hdu.header and hdu.header["IMAGE"] == "PRIMARY":
                self.set_fits_state(hdu.header)
                break
        return hdul

    def __str__(self):
        state = self.get_state()
        keys = ["data_shape", "pixelscale", "reference_imageij", "reference_imagexy"]
        if "zeropoint" in state:
            keys.append("zeropoint")
        if "note" in state:
            keys.append("note")
        return "\n".join(f"{key}: {state[key]}" for key in keys)

    def __repr__(self):
        state = self.get_state()
        return "\n".join(f"{key}: {state[key]}" for key in sorted(state.keys()))
