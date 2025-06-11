from typing import Optional, Union, Any

import torch
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from caskade import Module, Param, forward

from .. import AP_config
from ..utils.conversions.units import deg_to_arcsec
from .window import Window
from ..errors import SpecificationConflict, InvalidWindow, InvalidImage
from . import func

__all__ = ["Image", "Image_List"]


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

    default_crpix = (0.0, 0.0)
    default_crtan = (0.0, 0.0)
    default_crval = (0.0, 0.0)
    default_pixelscale = ((1.0, 0.0), (0.0, 1.0))

    def __init__(
        self,
        *,
        data: Optional[torch.Tensor] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        wcs: Optional[AstropyWCS] = None,
        filename: Optional[str] = None,
        identity: str = None,
        state: Optional[dict] = None,
        fits_state: Optional[dict] = None,
        name: Optional[str] = None,
        **kwargs: Any,
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
        if state is not None:
            self.set_state(state)
            return
        if fits_state is not None:
            self.set_fits_state(fits_state)
            return
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

            if "crpix" in kwargs or "crval" in kwargs:
                AP_config.ap_logger.warning(
                    "WCS crpix/crval set with supplied WCS, ignoring user supplied crpix/crval!"
                )
            kwargs["crval"] = wcs.wcs.crval
            kwargs["crpix"] = wcs.wcs.crpix

            if pixelscale is not None:
                AP_config.ap_logger.warning(
                    "WCS pixelscale set with supplied WCS, ignoring user supplied pixelscale!"
                )
            pixelscale = deg_to_arcsec * wcs.pixel_scale_matrix

        # set the data
        self.data = Param("data", data, units="flux")
        self.crval = Param("crval", kwargs.get("crval", self.default_crval), units="deg")
        self.crtan = Param("crtan", kwargs.get("crtan", self.default_crtan), units="arcsec")
        self.crpix = np.asarray(
            kwargs.get(
                "crpix",
                (
                    self.default_crpix
                    if self.data.value is None
                    else (self.data.shape[1] // 2, self.data.shape[0] // 2)
                ),
            ),
            dtype=int,
        )

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
        return Window(window=((0, 0), self.data.shape), crpix=self.crpix, image=self)

    @property
    def center(self):
        return self.pixel_to_plane(*(self.data.shape // 2))

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
            AP_config.ap_logger.warning(
                "Assuming diagonal pixelscale with the same value on both axes, please provide a full matrix to remove this message!"
            )
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
    def pixel_to_plane(self, i, j, crtan, pixelscale):
        return func.pixel_to_plane_linear(i, j, *self.crpix, pixelscale, *crtan)

    @forward
    def plane_to_pixel(self, x, y, crtan):
        return func.plane_to_pixel_linear(x, y, *self.crpix, self.pixelscale_inv, *crtan)

    @forward
    def plane_to_world(self, x, y, crval, crtan):
        return func.plane_to_world_gnomonic(x, y, *crval, *crtan)

    @forward
    def world_to_plane(self, ra, dec, crval, crtan):
        return func.world_to_plane_gnomonic(ra, dec, *crval, *crtan)

    @forward
    def world_to_pixel(self, ra, dec=None):
        """A wrapper which applies :meth:`world_to_plane` then
        :meth:`plane_to_pixel`, see those methods for further
        information.

        """
        if dec is None:
            ra, dec = ra[0], ra[1]
        return self.plane_to_pixel(*self.world_to_plane(ra, dec))

    @forward
    def pixel_to_world(self, i, j=None):
        """A wrapper which applies :meth:`pixel_to_plane` then
        :meth:`plane_to_world`, see those methods for further
        information.

        """
        if j is None:
            i, j = i[0], i[1]
        return self.plane_to_world(*self.pixel_to_plane(i, j))

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        copy_kwargs = {
            "data": torch.clone(self.data.value),
            "pixelscale": self.pixelscale.value,
            "crpix": self.crpix,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(**copy_kwargs)

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        copy_kwargs = {
            "data": torch.zeros_like(self.data.value),
            "pixelscale": self.pixelscale.value,
            "crpix": self.crpix,
            "crval": self.crval.value,
            "crtan": self.crtan.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
        }
        copy_kwargs.update(kwargs)
        return self.__class__(**copy_kwargs)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)
        if self.zeropoint is not None:
            self.zeropoint = self.zeropoint.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels, **kwargs):
        """Crop the image by the number of pixels given. This will crop
        the image in all four directions by the number of pixels given.

        given data shape (N, M) the new shape will be:

        crop - int: crop the same number of pixels on all sides. new shape (N - 2*crop, M - 2*crop)
        crop - (int, int): crop each dimension by the number of pixels given. new shape (N - 2*crop[1], M - 2*crop[0])
        crop - (int, int, int, int): crop each side by the number of pixels given assuming (x low, x high, y low, y high). new shape (N - crop[2] - crop[3], M - crop[0] - crop[1])
        """
        if isinstance(pixels, int) or len(pixels) == 1:  # same crop in all dimension
            crop = pixels if isinstance(pixels, int) else pixels[0]
            data = self.data.value[
                crop : self.data.shape[0] - crop,
                crop : self.data.shape[1] - crop,
            ]
            crpix = self.crpix - crop
        elif len(pixels) == 2:  # different crop in each dimension
            data = self.data.value[
                pixels[1] : self.data.shape[0] - pixels[1],
                pixels[0] : self.data.shape[1] - pixels[0],
            ]
            crpix = self.crpix - pixels
        elif len(pixels) == 4:  # different crop on all sides
            data = self.data.value[
                pixels[2] : self.data.shape[0] - pixels[3],
                pixels[0] : self.data.shape[1] - pixels[1],
            ]
            crpix = self.crpix - pixels[0::2]  # fixme
        else:
            raise ValueError(
                f"Invalid crop shape {pixels}, must be int, (int,), (int, int), or (int, int, int, int)!"
            )
        return self.copy(data=data, crpix=crpix, **kwargs)

    def flatten(self, attribute: str = "data") -> np.ndarray:
        return getattr(self, attribute).reshape(-1)

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
        if not isinstance(scale, int) and not (
            isinstance(scale, torch.Tensor) and scale.dtype is torch.int32
        ):
            raise SpecificationConflict(f"Reduce scale must be an integer! not {type(scale)}")
        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale

        data = (
            self.data.value[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .sum(axis=(1, 3))
        )
        pixelscale = self.pixelscale.value * scale
        crpix = (self.crpix + 0.5) / scale - 0.5
        return self.copy(
            data=data,
            pixelscale=pixelscale,
            crpix=crpix,
            **kwargs,
        )

    def get_state(self):
        state = {}
        state["type"] = self.__class__.__name__
        state["data"] = self.data.detach().cpu().tolist()
        state["crpix"] = self.crpix
        state["crtan"] = self.crtan.npvalue
        state["crval"] = self.crval.npvalue
        state["pixelscale"] = self.pixelscale.npvalue
        state["zeropoint"] = self.zeropoint
        state["identity"] = self.identity
        return state

    def set_state(self, state):
        self.data = state["data"]
        self.crpix = state["crpix"]
        self.crtan = state["crtan"]
        self.crval = state["crval"]
        self.pixelscale = state["pixelscale"]
        self.zeropoint = state["zeropoint"]
        self.identity = state["identity"]

    def get_fits_state(self):
        states = [{}]
        states[0]["DATA"] = self.data.detach().cpu().numpy()
        states[0]["HEADER"] = self.header.get_fits_state()
        states[0]["HEADER"]["IMAGE"] = "PRIMARY"
        return states

    def set_fits_state(self, states):
        for state in states:
            if state["HEADER"]["IMAGE"] == "PRIMARY":
                self.set_data(np.array(state["DATA"], dtype=np.float64), require_shape=False)
                self.header.set_fits_state(state["HEADER"])
                break

    def get_astropywcs(self, **kwargs):
        wargs = {
            "NAXIS": 2,
            "NAXIS1": self.pixel_shape[0].item(),
            "NAXIS2": self.pixel_shape[1].item(),
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": self.pixel_to_world(self.reference_imageij)[0].item(),
            "CRVAL2": self.pixel_to_world(self.reference_imageij)[1].item(),
            "CRPIX1": self.reference_imageij[0].item(),
            "CRPIX2": self.reference_imageij[1].item(),
            "CD1_1": self.pixelscale[0][0].item(),
            "CD1_2": self.pixelscale[0][1].item(),
            "CD2_1": self.pixelscale[1][0].item(),
            "CD2_2": self.pixelscale[1][1].item(),
        }
        wargs.update(kwargs)
        return AstropyWCS(wargs)

    def save(self, filename=None, overwrite=True):
        states = self.get_fits_state()
        img_list = [fits.PrimaryHDU(states[0]["DATA"], header=fits.Header(states[0]["HEADER"]))]
        for state in states[1:]:
            img_list.append(fits.ImageHDU(state["DATA"], header=fits.Header(state["HEADER"])))
        hdul = fits.HDUList(img_list)
        if filename is not None:
            hdul.writeto(filename, overwrite=overwrite)
        return hdul

    def load(self, filename):
        hdul = fits.open(filename)
        states = list({"DATA": hdu.data, "HEADER": hdu.header} for hdu in hdul)
        self.set_fits_state(states)

    @torch.no_grad()
    def get_indices(self, other: "Image"):
        origin_pix = torch.round(self.plane_to_pixel(other.pixel_to_plane(-0.5, -0.5)) + 0.5).int()
        new_origin_pix = torch.maximum(torch.zeros_like(origin_pix), origin_pix)

        end_pix = torch.round(
            self.plane_to_pixel(
                other.pixel_to_plane(other.data.shape[0] - 0.5, other.data.shape[1] - 0.5)
            )
            + 0.5
        ).int()
        new_end_pix = torch.minimum(self.data.shape, end_pix)
        return slice(new_origin_pix[1], new_end_pix[1]), slice(new_origin_pix[0], new_end_pix[0])

    def get_window(self, other: "Image", _indices=None, **kwargs):
        """Get a new image object which is a window of this image
        corresponding to the other image's window. This will return a
        new image object with the same properties as this one, but with
        the data cropped to the other image's window.

        """
        if not isinstance(other, Image):
            raise InvalidWindow("get_window only works with Image objects!")
        if _indices is None:
            indices = self.get_indices(other)
        else:
            indices = _indices
        new_img = self.copy(
            data=self.data.value[indices],
            crpix=self.crpix - np.array((indices[0].start, indices[1].start)),
            **kwargs,
        )
        return new_img

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img.data._value -= other[self].data.value
            return new_img
        else:
            new_img = self.copy()
            new_img.data._value -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other]
            new_img.data._value += other[self].data.value
            return new_img
        else:
            new_img = self.copy()
            new_img.data._value += other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image):
            self.data._value[self.get_indices(other)] += other.data.value[other.get_indices(self)]
        else:
            self.data._value += other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            self.data._value[self.get_indices(other)] -= other.data.value[other.get_indices(self)]
        else:
            self.data._value -= other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Image):
            return self.get_window(args[0])
        raise ValueError("Unrecognized Image getitem request!")


class Image_List(Module):
    def __init__(self, image_list):
        self.image_list = list(image_list)
        if not all(isinstance(image, Image) for image in self.image_list):
            raise InvalidImage(
                f"Image_List can only hold Image objects, not {tuple(type(image) for image in self.image_list)}"
            )

    @property
    def pixelscale(self):
        return tuple(image.pixelscale.value for image in self.image_list)

    @property
    def zeropoint(self):
        return tuple(image.zeropoint for image in self.image_list)

    @property
    def data(self):
        return tuple(image.data for image in self.image_list)

    @data.setter
    def data(self, data):
        for image, dat in zip(self.image_list, data):
            image.data = dat

    def copy(self):
        return self.__class__(
            tuple(image.copy() for image in self.image_list),
        )

    def blank_copy(self):
        return self.__class__(
            tuple(image.blank_copy() for image in self.image_list),
        )

    def get_window(self, other: "Image_List"):
        return self.__class__(
            tuple(image[win] for image, win in zip(self.image_list, other.image_list)),
        )

    def index(self, other):
        for i, image in enumerate(self.image_list):
            if other.identity == image.identity:
                return i
        else:
            raise ValueError("Could not find identity match between image list and input image")

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
        return torch.cat(tuple(image.flatten(attribute) for image in self.image_list))

    def __sub__(self, other):
        if isinstance(other, Image_List):
            new_list = []
            for other_image in other.image_list:
                i = self.index(other_image)
                self_image = self.image_list[i]
                new_list.append(self_image - other_image)
            return self.__class__(new_list)
        else:
            raise ValueError("Subtraction of Image_List only works with another Image_List object!")

    def __add__(self, other):
        if isinstance(other, Image_List):
            new_list = []
            for other_image in other.image_list:
                i = self.index(other_image)
                self_image = self.image_list[i]
                new_list.append(self_image + other_image)
            return self.__class__(new_list)
        else:
            raise ValueError("Addition of Image_List only works with another Image_List object!")

    def __isub__(self, other):
        if isinstance(other, Image_List):
            for other_image in other.image_list:
                i = self.index(other_image)
                self.image_list[i] -= other_image
        elif isinstance(other, Image):
            i = self.index(other)
            self.image_list[i] -= other
        else:
            raise ValueError("Subtraction of Image_List only works with another Image_List object!")
        return self

    def __iadd__(self, other):
        if isinstance(other, Image_List):
            for other_image in other.image_list:
                i = self.index(other_image)
                self.image_list[i] += other_image
        elif isinstance(other, Image):
            i = self.index(other)
            self.image_list[i] += other
        else:
            raise ValueError("Addition of Image_List only works with another Image_List object!")
        return self

    def save(self, filename=None, overwrite=True):
        raise NotImplementedError("Save/load not yet available for image lists")

    def load(self, filename):
        raise NotImplementedError("Save/load not yet available for image lists")

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Image_List):
            new_list = []
            for other_image in args[0].image_list:
                i = self.index(other_image)
                self_image = self.image_list[i]
                new_list.append(self_image.get_window(other_image))
            return self.__class__(new_list)
        raise ValueError("Unrecognized Image_List getitem request!")

    def __iter__(self):
        return (img for img in self.image_list)
