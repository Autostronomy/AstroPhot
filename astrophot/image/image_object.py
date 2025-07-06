from typing import Optional, Union, Any

import torch
import numpy as np
from astropy.wcs import WCS as AstropyWCS
from astropy.io import fits

from ..param import Module, Param, forward
from .. import AP_config
from ..utils.conversions.units import deg_to_arcsec
from .window import Window, WindowList
from ..errors import InvalidImage
from . import func

__all__ = ["Image", "ImageList"]


class Image(Module):
    """Core class to represent images with pixel values, pixel scale,
       and a window defining the spatial coordinates on the sky.
       It supports arithmetic operations with other image objects while preserving logical image boundaries.
       It also provides methods for determining the coordinate locations of pixels

    Parameters:
        data: the matrix of pixel values for the image
        pixelscale: the length of one side of a pixel in arcsec/pixel
        window: an AstroPhot Window object which defines the spatial coordinates on the sky
        filename: a filename from which to load the image.
        zeropoint: photometric zero point for converting from pixel flux to magnitude
        metadata: Any information the user wishes to associate with this image, stored in a python dictionary
        origin: The origin of the image in the coordinate system.
    """

    default_crpix = (0, 0)
    default_crtan = (0.0, 0.0)
    default_crval = (0.0, 0.0)
    default_pixelscale = ((1.0, 0.0), (0.0, 1.0))

    def __init__(
        self,
        *,
        data: Optional[torch.Tensor] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        crpix: Union[torch.Tensor, tuple] = (0, 0),
        crtan: Union[torch.Tensor, tuple] = (0.0, 0.0),
        crval: Union[torch.Tensor, tuple] = (0.0, 0.0),
        wcs: Optional[AstropyWCS] = None,
        filename: Optional[str] = None,
        identity: str = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize an instance of the APImage class.

        Parameters:
        -----------
        data : numpy.ndarray or None, optional
            The image data. Default is None.
        wcs : astropy.wcs.wcs.WCS or None, optional
            A WCS object which defines a coordinate system for the image. Note that AstroPhot only handles basic WCS conventions. It will use the WCS object to get `wcs.pixel_to_world(-0.5, -0.5)` to determine the position of the origin in world coordinates. It will also extract the `pixel_scale_matrix` to index pixels going forward.
        pixelscale : float or None, optional
            The physical scale of the pixels in the image, in units of arcseconds. Default is None.
        filename : str or None, optional
            The name of a file containing the image data. Default is None.
        zeropoint : float or None, optional
            The image's zeropoint, used for flux calibration. Default is None.

        """
        super().__init__(name=name)
        self.data = Param(
            "data", units="flux", dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.crval = Param(
            "crval", units="deg", dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.crtan = Param(
            "crtan", units="arcsec", dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self.crpix = Param(
            "crpix", units="pixel", dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

        if filename is not None:
            self.load(filename)
            return

        if identity is None:
            self.identity = id(self)
        else:
            self.identity = identity

        if wcs is not None:
            if wcs.wcs.ctype[0] != "RA---TAN":  # fixme handle sip
                AP_config.ap_logger.warning(
                    "Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot."
                )
            if wcs.wcs.ctype[1] != "DEC--TAN":
                AP_config.ap_logger.warning(
                    "Astropy WCS not tangent plane coordinate system! May not be compatible with AstroPhot."
                )

            crval = wcs.wcs.crval
            crpix = np.array(wcs.wcs.crpix) - 1  # handle FITS 1-indexing

            if pixelscale is not None:
                AP_config.ap_logger.warning(
                    "WCS pixelscale set with supplied WCS, ignoring user supplied pixelscale!"
                )
            pixelscale = deg_to_arcsec * wcs.pixel_scale_matrix

        # set the data
        self.data = data
        self.crval = crval
        self.crtan = crtan
        self.crpix = crpix

        self.pixelscale = pixelscale

        self.zeropoint = zeropoint

    @property
    def zeropoint(self):
        """The zeropoint of the image, which is used to convert from pixel flux to magnitude."""
        return self._zeropoint

    @zeropoint.setter
    def zeropoint(self, value):
        """Set the zeropoint of the image."""
        if value is None:
            self._zeropoint = None
        else:
            self._zeropoint = torch.as_tensor(
                value, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

    @property
    def window(self):
        return Window(window=((0, 0), self.data.shape[:2]), image=self)

    @property
    def center(self):
        shape = torch.as_tensor(
            self.data.shape[:2], dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        return self.pixel_to_plane(*((shape - 1) / 2))

    @property
    def shape(self):
        """The shape of the image data."""
        return self.data.shape

    @property
    def pixelscale(self):
        return self._pixelscale

    @pixelscale.setter
    def pixelscale(self, pixelscale):
        if pixelscale is None:
            pixelscale = self.default_pixelscale
        elif isinstance(pixelscale, (float, int)) or (
            isinstance(pixelscale, torch.Tensor) and pixelscale.numel() == 1
        ):
            pixelscale = ((pixelscale, 0.0), (0.0, pixelscale))
        self._pixelscale = torch.as_tensor(
            pixelscale, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        self._pixel_area = torch.linalg.det(self._pixelscale).abs()
        self._pixel_length = self._pixel_area.sqrt()
        self._pixelscale_inv = torch.linalg.inv(self._pixelscale)

    @property
    def pixel_area(self):
        """The area inside a pixel in arcsec^2"""
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
    def pixelscale_inv(self):
        """The inverse of the pixel scale matrix, which is used to
        transform tangent plane coordinates into pixel coordinates.

        """
        return self._pixelscale_inv

    @forward
    def pixel_to_plane(self, i, j, crpix, crtan):
        return func.pixel_to_plane_linear(i, j, *crpix, self.pixelscale, *crtan)

    @forward
    def plane_to_pixel(self, x, y, crpix, crtan):
        return func.plane_to_pixel_linear(x, y, *crpix, self.pixelscale_inv, *crtan)

    @forward
    def plane_to_world(self, x, y, crval, crtan):
        return func.plane_to_world_gnomonic(x, y, *crval, *crtan)

    @forward
    def world_to_plane(self, ra, dec, crval, crtan):
        return func.world_to_plane_gnomonic(ra, dec, *crval, *crtan)

    @forward
    def world_to_pixel(self, ra, dec):
        """A wrapper which applies :meth:`world_to_plane` then
        :meth:`plane_to_pixel`, see those methods for further
        information.

        """
        return self.plane_to_pixel(*self.world_to_plane(ra, dec))

    @forward
    def pixel_to_world(self, i, j):
        """A wrapper which applies :meth:`pixel_to_plane` then
        :meth:`plane_to_world`, see those methods for further
        information.

        """
        return self.plane_to_world(*self.pixel_to_plane(i, j))

    def pixel_center_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, centered on the pixel grid."""
        return func.pixel_center_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_corner_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, with corners at the pixel grid."""
        return func.pixel_corner_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_simpsons_meshgrid(self):
        """Get a meshgrid of pixel coordinates in the image, with Simpson's rule sampling."""
        return func.pixel_simpsons_meshgrid(self.shape, AP_config.ap_dtype, AP_config.ap_device)

    def pixel_quad_meshgrid(self, order=3):
        """Get a meshgrid of pixel coordinates in the image, with quadrature sampling."""
        return func.pixel_quad_meshgrid(
            self.shape, AP_config.ap_dtype, AP_config.ap_device, order=order
        )

    @forward
    def coordinate_center_meshgrid(self):
        """Get a meshgrid of coordinate locations in the image, centered on the pixel grid."""
        i, j = self.pixel_center_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_corner_meshgrid(self):
        """Get a meshgrid of coordinate locations in the image, with corners at the pixel grid."""
        i, j = self.pixel_corner_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_simpsons_meshgrid(self):
        """Get a meshgrid of coordinate locations in the image, with Simpson's rule sampling."""
        i, j = self.pixel_simpsons_meshgrid()
        return self.pixel_to_plane(i, j)

    @forward
    def coordinate_quad_meshgrid(self, order=3):
        """Get a meshgrid of coordinate locations in the image, with quadrature sampling."""
        i, j, _ = self.pixel_quad_meshgrid(order=order)
        return self.pixel_to_plane(i, j)

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        kwargs = {
            "data": torch.clone(self.data.value.detach()),
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name,
            **kwargs,
        }
        return self.__class__(**kwargs)

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        kwargs = {
            "data": torch.zeros_like(self.data.value),
            "pixelscale": self.pixelscale,
            "crpix": self.crpix.value,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            "name": self.name,
            **kwargs,
        }
        return self.__class__(**kwargs)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint = self.zeropoint.to(dtype=dtype, device=device)
        return self

    def flatten(self, attribute: str = "data") -> torch.Tensor:
        if attribute in self.children:
            return getattr(self, attribute).value.reshape(-1)
        return getattr(self, attribute).reshape(-1)

    def fits_info(self):
        return {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": self.crval.value[0].item(),
            "CRVAL2": self.crval.value[1].item(),
            "CRPIX1": self.crpix.value[0].item(),
            "CRPIX2": self.crpix.value[1].item(),
            "CRTAN1": self.crtan.value[0].item(),
            "CRTAN2": self.crtan.value[1].item(),
            "CD1_1": self.pixelscale[0][0].item(),
            "CD1_2": self.pixelscale[0][1].item(),
            "CD2_1": self.pixelscale[1][0].item(),
            "CD2_2": self.pixelscale[1][1].item(),
            "MAGZP": self.zeropoint.item() if self.zeropoint is not None else -999,
            "IDNTY": self.identity,
        }

    def fits_images(self):
        return [
            fits.PrimaryHDU(self.data.value.cpu().numpy(), header=fits.Header(self.fits_info()))
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

    def load(self, filename: str):
        """Load an image from a FITS file. This will load the primary HDU
        and set the data, pixelscale, crpix, crval, and crtan attributes
        accordingly. If the WCS is not tangent plane, it will warn the user.

        """
        hdulist = fits.open(filename)
        self.data = torch.as_tensor(
            np.array(hdulist[0].data, dtype=np.float64),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        self.pixelscale = (
            (hdulist[0].header["CD1_1"], hdulist[0].header["CD1_2"]),
            (hdulist[0].header["CD2_1"], hdulist[0].header["CD2_2"]),
        )
        self.crpix = (hdulist[0].header["CRPIX1"], hdulist[0].header["CRPIX2"])
        self.crval = (hdulist[0].header["CRVAL1"], hdulist[0].header["CRVAL2"])
        if "CRTAN1" in hdulist[0].header and "CRTAN2" in hdulist[0].header:
            self.crtan = (hdulist[0].header["CRTAN1"], hdulist[0].header["CRTAN2"])
        else:
            self.crtan = (0.0, 0.0)
        if "MAGZP" in hdulist[0].header and hdulist[0].header["MAGZP"] > -998:
            self.zeropoint = hdulist[0].header["MAGZP"]
        else:
            self.zeropoint = None
        self.identity = hdulist[0].header.get("IDNTY", str(id(self)))
        return hdulist

    def corners(self):
        pixel_lowleft = torch.tensor(
            (-0.5, -0.5), dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        pixel_lowright = torch.tensor(
            (self.data.shape[0] - 0.5, -0.5), dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        pixel_upleft = torch.tensor(
            (-0.5, self.data.shape[1] - 0.5), dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        pixel_upright = torch.tensor(
            (self.data.shape[0] - 0.5, self.data.shape[1] - 0.5),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        lowleft = self.pixel_to_plane(*pixel_lowleft)
        lowright = self.pixel_to_plane(*pixel_lowright)
        upleft = self.pixel_to_plane(*pixel_upleft)
        upright = self.pixel_to_plane(*pixel_upright)
        return (lowleft, lowright, upright, upleft)

    @torch.no_grad()
    def get_indices(self, other: Window):
        if other.image == self:
            return slice(max(0, other.i_low), min(self.shape[0], other.i_high)), slice(
                max(0, other.j_low), min(self.shape[1], other.j_high)
            )
        shift = np.round(self.crpix.npvalue - other.crpix.npvalue).astype(int)
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
        # origin_pix = torch.tensor(
        #     (-0.5, -0.5), dtype=AP_config.ap_dtype, device=AP_config.ap_device
        # )
        # origin_pix = self.plane_to_pixel(*other.pixel_to_plane(*origin_pix))
        # origin_pix = torch.round(torch.stack(origin_pix) + 0.5).int()
        # new_origin_pix = torch.maximum(torch.zeros_like(origin_pix), origin_pix)

        # end_pix = torch.tensor(
        #     (other.data.shape[0] - 0.5, other.data.shape[1] - 0.5),
        #     dtype=AP_config.ap_dtype,
        #     device=AP_config.ap_device,
        # )
        # end_pix = self.plane_to_pixel(*other.pixel_to_plane(*end_pix))
        # end_pix = torch.round(torch.stack(end_pix) + 0.5).int()
        # shape = torch.tensor(self.data.shape[:2], dtype=torch.int32, device=AP_config.ap_device)
        # new_end_pix = torch.minimum(shape, end_pix)
        # return slice(new_origin_pix[0], new_end_pix[0]), slice(new_origin_pix[1], new_end_pix[1])

    def get_window(self, other: Union[Window, "Image"], _indices=None, **kwargs):
        """Get a new image object which is a window of this image
        corresponding to the other image's window. This will return a
        new image object with the same properties as this one, but with
        the data cropped to the other image's window.

        """
        if _indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        else:
            indices = _indices
        new_img = self.copy(
            data=self.data.value[indices],
            crpix=self.crpix.value
            - torch.tensor(
                (indices[0].start, indices[1].start),
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            ),
            **kwargs,
        )
        return new_img

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img.data._value = new_img.data._value - other[self].data.value
            return new_img
        else:
            new_img = self.copy()
            new_img.data._value = new_img.data._value - other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img.data._value = new_img.data._value + other[self].data.value
            return new_img
        else:
            new_img = self.copy()
            new_img.data._value = new_img.data._value + other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image):
            self.data._value[self.get_indices(other.window)] += other.data.value[
                other.get_indices(self.window)
            ]
        else:
            self.data._value = self.data._value + other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            self.data._value[self.get_indices(other.window)] -= other.data.value[
                other.get_indices(self.window)
            ]
        else:
            self.data._value = self.data._value - other
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
    def pixelscale(self):
        return tuple(image.pixelscale for image in self.images)

    @property
    def zeropoint(self):
        return tuple(image.zeropoint for image in self.images)

    @property
    def data(self):
        return tuple(image.data.value for image in self.images)

    @data.setter
    def data(self, data):
        for image, dat in zip(self.images, data):
            image.data = dat

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
            raise ValueError("Could not find identity match between image list and input image")

    def match_indices(self, other: "ImageList"):
        """Match the indices of the images in this list with those in another Image_List."""
        indices = []
        for other_image in other.images:
            try:
                i = self.index(other_image)
            except ValueError:
                continue
            indices.append(i)
        return indices

    def to(self, dtype=None, device=None):
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)
        return self

    def crop(self, *pixels):
        raise NotImplementedError("Crop function not available for Image_List object")

    def flatten(self, attribute="data"):
        return torch.cat(tuple(image.flatten(attribute) for image in self.images))

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
                i = self.index(other_image)
                self_image = self.images[i]
                new_list.append(self_image + other_image)
            return self.__class__(new_list)
        else:
            raise ValueError("Addition of Image_List only works with another Image_List object!")

    def __isub__(self, other):
        if isinstance(other, ImageList):
            for other_image in other.images:
                i = self.index(other_image)
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
                i = self.index(other_image)
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
        super().__getitem__(*args)

    def __iter__(self):
        return (img for img in self.images)
