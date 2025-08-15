from typing import Optional, Tuple, Union

import torch
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.io import fits

from ..param import Module, Param, forward
from .. import config
from ..backend_obj import backend, ArrayLike
from ..utils.conversions.units import deg_to_arcsec, arcsec_to_deg
from .window import Window, WindowList
from ..errors import InvalidImage, SpecificationConflict

# from .base import BaseImage
from . import func

__all__ = ["Image", "ImageList"]


class Image(Module):
    """Core class to represent images with pixel values, pixel scale,
    and a window defining the spatial coordinates on the sky. It supports
    arithmetic operations with other image objects while preserving logical
    image boundaries. It also provides methods for determining the coordinate
    locations of pixels

    **Args:**
    -  `data`: The image data as a tensor of pixel values. If not provided, a tensor of zeros will be created.
    -  `zeropoint`: The zeropoint of the image, which is used to convert from pixel flux to magnitude.
    -  `crpix`: The reference pixel coordinates in the image, which is used to convert from pixel coordinates to tangent plane coordinates.
    -  `pixelscale`: The side length of a pixel, used to create a simple diagonal CD matrix.
    -  `wcs`: An optional Astropy WCS object to initialize the image.
    -  `filename`: The filename to load the image from. If provided, the image will be loaded from the file.
    -  `hduext`: The HDU extension to load from the FITS file specified in `filename`.
    -  `identity`: An optional identity string for the image.

    these parameters are added to the optimization model:

    **Parameters:**
    -  `crval`: The reference coordinate of the image in degrees [RA, DEC].
    -  `crtan`: The tangent plane coordinate of the image in arcseconds [x, y].
    -  `CD`: The coordinate transformation matrix in arcseconds/pixel.
    """

    default_CD = ((1.0, 0.0), (0.0, 1.0))
    expect_ctype = (("RA---TAN",), ("DEC--TAN",))
    base_scale = 1.0

    def __init__(
        self,
        *,
        data: Optional[ArrayLike] = None,
        CD: Optional[Union[float, ArrayLike]] = None,
        zeropoint: Optional[Union[float, ArrayLike]] = None,
        crpix: Union[ArrayLike, tuple] = (0.0, 0.0),
        crtan: Union[ArrayLike, tuple] = (0.0, 0.0),
        crval: Union[ArrayLike, tuple] = (0.0, 0.0),
        pixelscale: Optional[Union[ArrayLike, float]] = None,
        wcs: Optional[AstropyWCS] = None,
        filename: Optional[str] = None,
        hduext: int = 0,
        identity: str = None,
        name: Optional[str] = None,
        _data: Optional[ArrayLike] = None,
    ):
        super().__init__(name=name)
        if _data is None:
            self.data = data  # units: flux
        else:
            self._data = _data
        self.crval = Param(
            "crval", shape=(2,), units="deg", dtype=config.DTYPE, device=config.DEVICE
        )
        self.crtan = Param(
            "crtan",
            crtan,
            shape=(2,),
            units="arcsec",
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        self.CD = Param(
            "CD",
            shape=(2, 2),
            units="arcsec/pixel",
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        self.zeropoint = zeropoint

        if filename is not None:
            self.load(filename, hduext=hduext)
            return

        if identity is None:
            self.identity = id(self)
        else:
            self.identity = identity

        if wcs is not None:
            if wcs.wcs.ctype[0] not in self.expect_ctype[0]:
                config.logger.warning(
                    "Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot."
                )
            if wcs.wcs.ctype[1] not in self.expect_ctype[1]:
                config.logger.warning(
                    "Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot."
                )

            crval = wcs.wcs.crval
            crpix = np.array(wcs.wcs.crpix)[::-1] - 1  # handle FITS 1-indexing

            if CD is not None:
                config.logger.warning("WCS CD set with supplied WCS, ignoring user supplied CD!")
            CD = deg_to_arcsec * wcs.pixel_scale_matrix

        # set the data
        self.crval = crval
        self.crpix = crpix

        if isinstance(CD, (float, int)):
            CD = np.array([[CD, 0.0], [0.0, CD]], dtype=np.float64)
        elif CD is None and pixelscale is not None:
            CD = np.array([[pixelscale, 0.0], [0.0, pixelscale]], dtype=np.float64)
        elif CD is None:
            CD = self.default_CD
        self.CD = CD

    @property
    def data(self):
        """The image data, which is a tensor of pixel values."""
        return self._data

    @data.setter
    def data(self, value: Optional[ArrayLike]):
        """Set the image data. If value is None, the data is initialized to an empty tensor."""
        if value is None:
            self._data = backend.empty((0, 0), dtype=config.DTYPE, device=config.DEVICE)
        else:
            # Transpose since pytorch uses (j, i) indexing when (i, j) is more natural for coordinates
            self._data = backend.transpose(
                backend.as_array(value, dtype=config.DTYPE, device=config.DEVICE), 1, 0
            )

    @property
    def crpix(self) -> np.ndarray:
        """The reference pixel coordinates in the image, which is used to convert from pixel coordinates to tangent plane coordinates."""
        return self._crpix

    @crpix.setter
    def crpix(self, value: Union[ArrayLike, tuple]):
        self._crpix = np.asarray(value, dtype=np.float64)

    @property
    def zeropoint(self) -> ArrayLike:
        """The zeropoint of the image, which is used to convert from pixel flux to magnitude."""
        return self._zeropoint

    @zeropoint.setter
    def zeropoint(self, value):
        """Set the zeropoint of the image."""
        if value is None:
            self._zeropoint = None
        else:
            self._zeropoint = backend.as_array(value, dtype=config.DTYPE, device=config.DEVICE)

    @property
    def window(self) -> Window:
        return Window(window=((0, 0), self.data.shape[:2]), image=self)

    @property
    def center(self):
        shape = backend.as_array(self.data.shape[:2], dtype=config.DTYPE, device=config.DEVICE)
        return backend.stack(self.pixel_to_plane(*((shape - 1) / 2)))

    @property
    def shape(self):
        """The shape of the image data."""
        return self.data.shape

    @property
    @forward
    def pixel_area(self, CD):
        """The area inside a pixel in arcsec^2"""
        return backend.abs(backend.linalg.det(CD))

    @property
    @forward
    def pixelscale(self):
        """The approximate side length of a pixel, which is just
        sqrt(pixel_area). For square pixels this is the actual pixel
        length, for rectangular pixels it is a kind of average.

        The pixelscale is not used for exact calculations
        and instead sets a size scale within an image.

        """
        return backend.sqrt(self.pixel_area)

    @forward
    def pixel_to_plane(
        self,
        i: ArrayLike,
        j: ArrayLike,
        crtan: ArrayLike,
        CD: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        return func.pixel_to_plane_linear(i, j, *self.crpix, CD, *crtan)

    @forward
    def plane_to_pixel(
        self,
        x: ArrayLike,
        y: ArrayLike,
        crtan: ArrayLike,
        CD: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        return func.plane_to_pixel_linear(x, y, *self.crpix, CD, *crtan)

    @forward
    def plane_to_world(
        self, x: ArrayLike, y: ArrayLike, crval: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return func.plane_to_world_gnomonic(x, y, *crval)

    @forward
    def world_to_plane(
        self, ra: ArrayLike, dec: ArrayLike, crval: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        return func.world_to_plane_gnomonic(ra, dec, *crval)

    @forward
    def world_to_pixel(self, ra: ArrayLike, dec: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """A wrapper which applies :meth:`world_to_plane` then
        :meth:`plane_to_pixel`, see those methods for further
        information.

        """
        return self.plane_to_pixel(*self.world_to_plane(ra, dec))

    @forward
    def pixel_to_world(self, i: ArrayLike, j: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """A wrapper which applies :meth:`pixel_to_plane` then
        :meth:`plane_to_world`, see those methods for further
        information.

        """
        return self.plane_to_world(*self.pixel_to_plane(i, j))

    def pixel_center_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, centered on the pixel grid."""
        return func.pixel_center_meshgrid(self.shape, config.DTYPE, config.DEVICE)

    def pixel_corner_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with corners at the pixel grid."""
        return func.pixel_corner_meshgrid(self.shape, config.DTYPE, config.DEVICE)

    def pixel_simpsons_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with Simpson's rule sampling."""
        return func.pixel_simpsons_meshgrid(self.shape, config.DTYPE, config.DEVICE)

    def pixel_quad_meshgrid(self, order=3) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with quadrature sampling."""
        return func.pixel_quad_meshgrid(self.shape, config.DTYPE, config.DEVICE, order=order)

    @forward
    def coordinate_center_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of coordinate locations in the image, centered on the pixel grid."""
        i, j = self.pixel_center_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_corner_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of coordinate locations in the image, with corners at the pixel grid."""
        i, j = self.pixel_corner_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_simpsons_meshgrid(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of coordinate locations in the image, with Simpson's rule sampling."""
        i, j = self.pixel_simpsons_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_quad_meshgrid(self, order=3) -> Tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of coordinate locations in the image, with quadrature sampling."""
        i, j, _ = self.pixel_quad_meshgrid(order=order)
        return self.pixel_to_plane(i, j)

    def copy_kwargs(self, **kwargs) -> dict:
        kwargs = {
            "_data": backend.copy(self.data),
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name,
            **kwargs,
        }
        return kwargs

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return self.__class__(**self.copy_kwargs(**kwargs))

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        kwargs = {
            "_data": backend.zeros_like(self.data),
            **kwargs,
        }
        return self.copy(**kwargs)

    def crop(self, pixels: Union[int, Tuple[int, int], Tuple[int, int, int, int]], **kwargs):
        """Crop the image by the number of pixels given. This will crop
        the image in all four directions by the number of pixels given.

        given data shape (N, M) the new shape will be:

        crop - int: crop the same number of pixels on all sides. new shape (N - 2*crop, M - 2*crop)
        crop - (int, int): crop each dimension by the number of pixels given. new shape (N - 2*crop[1], M - 2*crop[0])
        crop - (int, int, int, int): crop each side by the number of pixels given assuming (x low, x high, y low, y high). new shape (N - crop[2] - crop[3], M - crop[0] - crop[1])
        """
        if isinstance(pixels, int):
            data = self.data[
                pixels : self.data.shape[0] - pixels,
                pixels : self.data.shape[1] - pixels,
            ]
            crpix = self.crpix - pixels
        elif len(pixels) == 1:  # same crop in all dimension
            crop = pixels if isinstance(pixels, int) else pixels[0]
            data = self.data[
                crop : self.data.shape[0] - crop,
                crop : self.data.shape[1] - crop,
            ]
            crpix = self.crpix - crop
        elif len(pixels) == 2:  # different crop in each dimension
            data = self.data[
                pixels[0] : self.data.shape[0] - pixels[0],
                pixels[1] : self.data.shape[1] - pixels[1],
            ]
            crpix = self.crpix - pixels
        elif len(pixels) == 4:  # different crop on all sides
            data = self.data[
                pixels[0] : self.data.shape[0] - pixels[1],
                pixels[2] : self.data.shape[1] - pixels[3],
            ]
            crpix = self.crpix - pixels[0::2]
        else:
            raise ValueError(
                f"Invalid crop shape {pixels}, must be (int,), (int, int), or (int, int, int, int)!"
            )
        return self.copy(_data=data, crpix=crpix, **kwargs)

    def reduce(self, scale: int, **kwargs):
        """This operation will downsample an image by the factor given. If
        scale = 2 then 2x2 blocks of pixels will be summed together to
        form individual larger pixels. A new image object will be
        returned with the appropriate pixelscale and data tensor. Note
        that the window does not change in this operation since the
        pixels are condensed, but the pixel size is increased
        correspondingly.

        **Args:**
        -  `scale` (int): The scale factor by which to reduce the image.
        """
        if not isinstance(scale, int) and not (
            isinstance(scale, ArrayLike) and scale.dtype is backend.int32
        ):
            raise SpecificationConflict(f"Reduce scale must be an integer! not {type(scale)}")
        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        data = self.data[: MS * scale, : NS * scale].reshape(MS, scale, NS, scale).sum(axis=(1, 3))
        CD = self.CD.value * scale
        crpix = (self.crpix + 0.5) / scale - 0.5
        return self.copy(
            _data=data,
            CD=CD,
            crpix=crpix,
            **kwargs,
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = config.DTYPE
        if device is None:
            device = config.DEVICE
        super().to(dtype=dtype, device=device)
        self._data = backend.to(self._data, dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint = backend.to(self.zeropoint, dtype=dtype, device=device)
        return self

    def flatten(self, attribute: str = "data") -> ArrayLike:
        return backend.flatten(getattr(self, attribute), end_dim=1)

    def fits_info(self) -> dict:
        return {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": self.crval.value[0].item(),
            "CRVAL2": self.crval.value[1].item(),
            "CRPIX1": self.crpix[0] + 1,
            "CRPIX2": self.crpix[1] + 1,
            "CRTAN1": self.crtan.value[0].item(),
            "CRTAN2": self.crtan.value[1].item(),
            "CD1_1": self.CD.value[0][0].item() * arcsec_to_deg,
            "CD1_2": self.CD.value[0][1].item() * arcsec_to_deg,
            "CD2_1": self.CD.value[1][0].item() * arcsec_to_deg,
            "CD2_2": self.CD.value[1][1].item() * arcsec_to_deg,
            "MAGZP": self.zeropoint.item() if self.zeropoint is not None else -999,
            "IDNTY": self.identity,
        }

    def fits_images(self):
        return [
            fits.PrimaryHDU(
                backend.to_numpy(backend.transpose(self.data, 1, 0)),
                header=fits.Header(self.fits_info()),
            )
        ]

    def get_astropywcs(self, **kwargs):
        kwargs = {
            "NAXIS": 2,
            "NAXIS1": self.shape[0].item(),
            "NAXIS2": self.shape[1].item(),
            **self.fits_info(),
            **kwargs,
        }
        return AstropyWCS(kwargs)

    def save(self, filename: str):
        hdulist = fits.HDUList(self.fits_images())
        hdulist.writeto(filename, overwrite=True)

    def load(self, filename: str, hduext: int = 0):
        """Load an image from a FITS file. This will load the primary HDU
        and set the data, CD, crpix, crval, and crtan attributes
        accordingly. If the WCS is not tangent plane, it will warn the user.

        """
        hdulist = fits.open(filename)
        self.data = np.array(hdulist[hduext].data, dtype=np.float64)

        self.CD = (
            np.array(
                (
                    (hdulist[hduext].header["CD1_1"], hdulist[hduext].header["CD1_2"]),
                    (hdulist[hduext].header["CD2_1"], hdulist[hduext].header["CD2_2"]),
                ),
                dtype=np.float64,
            )
            * deg_to_arcsec
        )
        self.crpix = (hdulist[hduext].header["CRPIX1"] - 1, hdulist[hduext].header["CRPIX2"] - 1)
        self.crval = (hdulist[hduext].header["CRVAL1"], hdulist[hduext].header["CRVAL2"])
        if "CRTAN1" in hdulist[hduext].header and "CRTAN2" in hdulist[hduext].header:
            self.crtan = (hdulist[hduext].header["CRTAN1"], hdulist[hduext].header["CRTAN2"])
        if "MAGZP" in hdulist[hduext].header and hdulist[hduext].header["MAGZP"] > -998:
            self.zeropoint = hdulist[hduext].header["MAGZP"]
        self.identity = hdulist[hduext].header.get("IDNTY", str(id(self)))
        return hdulist

    def corners(
        self,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        pixel_lowleft = backend.make_array((-0.5, -0.5), dtype=config.DTYPE, device=config.DEVICE)
        pixel_lowright = backend.make_array(
            (self.data.shape[0] - 0.5, -0.5), dtype=config.DTYPE, device=config.DEVICE
        )
        pixel_upleft = backend.make_array(
            (-0.5, self.data.shape[1] - 0.5), dtype=config.DTYPE, device=config.DEVICE
        )
        pixel_upright = backend.make_array(
            (self.data.shape[0] - 0.5, self.data.shape[1] - 0.5),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        lowleft = self.pixel_to_plane(*pixel_lowleft)
        lowright = self.pixel_to_plane(*pixel_lowright)
        upleft = self.pixel_to_plane(*pixel_upleft)
        upright = self.pixel_to_plane(*pixel_upright)
        return (lowleft, lowright, upright, upleft)

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

    def get_window(self, other: Union[Window, "Image"], indices=None, **kwargs):
        """Get a new image object which is a window of this image
        corresponding to the other image's window. This will return a
        new image object with the same properties as this one, but with
        the data cropped to the other image's window.

        """
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        new_img = self.copy(
            _data=self.data[indices],
            crpix=self.crpix - np.array((indices[0].start, indices[1].start)),
            **kwargs,
        )
        return new_img

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img._data = new_img.data - other[self].data
            return new_img
        else:
            new_img = self.copy()
            new_img._data = new_img.data - other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img._data = new_img.data + other[self].data
            return new_img
        else:
            new_img = self.copy()
            new_img._data = new_img.data + other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image):
            self._data = backend.add_at_indices(
                self._data,
                self.get_indices(other.window),
                other.data[other.get_indices(self.window)],
            )
        else:
            self._data = self.data + other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            self._data = backend.add_at_indices(
                self._data,
                self.get_indices(other.window),
                -other.data[other.get_indices(self.window)],
            )
        else:
            self._data = self.data - other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], (Image, Window)):
            return self.get_window(args[0])
        return super().__getitem__(*args)


class ImageList(Module):
    def __init__(self, images, name=None):
        super().__init__(name=name)
        self.images = list(images)
        if not all(isinstance(image, Image) for image in self.images):
            raise InvalidImage(
                f"Image_List can only hold Image objects, not {tuple(type(image) for image in self.images)}"
            )

    @property
    def data(self):
        return tuple(image.data for image in self.images)

    def copy(self):
        return self.__class__(
            tuple(image.copy() for image in self.images),
        )

    def blank_copy(self):
        return self.__class__(
            tuple(image.blank_copy() for image in self.images),
        )

    def get_window(self, other: "ImageList"):
        return self.__class__(
            tuple(image[win] for image, win in zip(self.images, other.images)),
        )

    def index(self, other: Image):
        for i, image in enumerate(self.images):
            if other.identity == image.identity:
                return i
        else:
            raise IndexError(
                f"Could not find identity match between image list {self.name} and input image {other.name}"
            )

    def match_indices(self, other: "ImageList"):
        """Match the indices of the images in this list with those in another Image_List."""
        indices = []
        for other_image in other.images:
            try:
                i = self.index(other_image)
            except IndexError:
                continue
            indices.append(i)
        return indices

    def to(self, dtype=None, device=None):
        if dtype is not None:
            dtype = config.DTYPE
        if device is not None:
            device = config.DEVICE
        super().to(dtype=dtype, device=device)
        return self

    def flatten(self, attribute: str = "data") -> ArrayLike:
        return backend.concatenate(tuple(image.flatten(attribute) for image in self.images))

    def __sub__(self, other):
        if isinstance(other, ImageList):
            new_list = []
            for other_image in other.images:
                i = self.index(other_image)
                self_image = self.images[i]
                new_list.append(self_image - other_image)
            return self.__class__(new_list)
        else:
            raise ValueError("Subtraction of Image_List only works with another Image_List object!")

    def __add__(self, other):
        if isinstance(other, ImageList):
            new_list = []
            for other_image in other.images:
                try:
                    i = self.index(other_image)
                except IndexError:
                    continue
                self_image = self.images[i]
                new_list.append(self_image + other_image)
            return self.__class__(new_list)
        else:
            raise ValueError("Addition of Image_List only works with another Image_List object!")

    def __isub__(self, other):
        if isinstance(other, ImageList):
            for other_image in other.images:
                try:
                    i = self.index(other_image)
                except IndexError:
                    continue
                self.images[i] -= other_image
        elif isinstance(other, Image):
            i = self.index(other)
            self.images[i] -= other
        else:
            raise ValueError("Subtraction of Image_List only works with another Image_List object!")
        return self

    def __iadd__(self, other):
        if isinstance(other, ImageList):
            for other_image in other.images:
                try:
                    i = self.index(other_image)
                except IndexError:
                    continue
                self.images[i] += other_image
        elif isinstance(other, Image):
            i = self.index(other)
            self.images[i] += other
        else:
            raise ValueError("Addition of Image_List only works with another Image_List object!")
        return self

    def __getitem__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], ImageList):
                new_list = []
                for other_image in args[0].images:
                    i = self.index(other_image)
                    new_list.append(self.images[i].get_window(other_image))
                return self.__class__(new_list)
            elif isinstance(args[0], WindowList):
                new_list = []
                for other_window in args[0].windows:
                    i = self.index(other_window.image)
                    new_list.append(self.images[i].get_window(other_window))
                return self.__class__(new_list)
            elif isinstance(args[0], Image):
                i = self.index(args[0])
                return self.images[i].get_window(args[0])
            elif isinstance(args[0], Window):
                i = self.index(args[0].image)
                return self.images[i].get_window(args[0])
            elif isinstance(args[0], int):
                return self.images[args[0]]
        super().__getitem__(*args)

    def __iter__(self):
        return (img for img in self.images)
