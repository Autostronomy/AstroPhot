from typing import Optional, Union, Any, Sequence, Tuple
from copy import deepcopy

import torch
from torch.nn.functional import pad
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .window_object import Window, Window_List
from .image_header import Image_Header
from .. import AP_config

__all__ = ["Image", "Image_List"]


class Image(object):
    """Core class to represent images with pixel values, pixel scale,
       and a window defining the spatial coordinates on the sky.
       It supports arithmetic operations with other image objects while preserving logical image boundaries.
       It also provides methods for determining the coordinate locations of pixels

    Parameters:
        data: the matrix of pixel values for the image
        pixelscale: the length of one side of a pixel in arcsec/pixel
        window: an AutoPhot Window object which defines the spatial cooridnates on the sky
        filename: a filename from which to load the image.
        zeropoint: photometric zero point for converting from pixel flux to magnitude
        note: a note about this image if any
        origin: The origin of the image in the coordinate system.
    """

    def __init__(
        self,
        data: Optional[Union[torch.Tensor]] = None,
        header: Optional[Image_Header] = None,
        wcs: Optional["astropy.wcs.wcs.WCS"] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
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
        data : numpy.ndarray or None, optional
            The image data. Default is None.
        wcs : astropy.wcs.wcs.WCS or None, optional
            A WCS object which defines a coordinate system for the image. Note that AutoPhot only handles basic WCS conventions. It will use the WCS object to get `wcs.pixel_to_world(-0.5, -0.5)` to determine the position of the origin in world coordinates. It will also extract the `pixel_scale_matrix` to index pixels going forward.
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
        self._data = None

        if header is None:
            self.header = Image_Header(
                data_shape=None if data is None else data.shape,
                pixelscale=pixelscale,
                wcs=wcs,
                window=window,
                filename=filename,
                zeropoint=zeropoint,
                note=note,
                origin=origin,
                center=center,
                identity=identity,
                **kwargs,
            )
        else:
            self.header = header

        if filename is not None:
            self.load(filename)
            return

        # set the data
        self.data = data
        self.to()

    @property
    def north(self):
        return self.header.north

    @property
    def pixel_area(self):
        return self.header.pixel_area

    @property
    def pixel_length(self):
        return self.header.pixel_length

    def pixel_to_world(self, pixel_coordinate, internal_transpose=False):
        return self.header.pixel_to_world(
            pixel_coordinate, internal_transpose=internal_transpose
        )

    def world_to_pixel(self, world_coordinate, unsqueeze_origin=False):
        return self.header.world_to_pixel(world_coordinate, unsqueeze_origin)

    def pixel_to_world_delta(self, pixel_coordinate):
        return self.header.pixel_to_world_delta(pixel_coordinate)

    def world_to_pixel_delta(self, world_coordinate):
        return self.header.world_to_pixel_delta(world_coordinate)

    @property
    def origin(self) -> torch.Tensor:
        """
        Returns the origin (bottom-left corner) of the image window.

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the origin.
        """
        return self.header.window.origin

    @property
    def shape(self) -> torch.Tensor:
        """
        Returns the shape (size) of the image window.

        Returns:
                torch.Tensor: A 1D tensor of shape (2,) containing the (width, height) of the window in pixels.
        """
        return self.header.window.shape

    @property
    def center(self) -> torch.Tensor:
        """
        Returns the center of the image window.

        Returns:
            torch.Tensor: A 1D tensor of shape (2,) containing the (x, y) coordinates of the center.
        """
        return self.header.window.center

    @property
    def window(self):
        return self.header.window

    @property
    def pixelscale(self):
        return self.header.pixelscale

    @property
    def zeropoint(self):
        return self.header.zeropoint

    @property
    def note(self):
        return self.header.note

    @property
    def identity(self):
        return self.header.identity

    @property
    def data(self) -> torch.Tensor:
        """
        Returns the image data.
        """
        return self._data

    @data.setter
    def data(self, data) -> None:
        """Set the image data."""
        self.set_data(data)

    def set_data(
        self, data: Union[torch.Tensor, np.ndarray], require_shape: bool = True
    ):
        """
        Set the image data.

        Args:
            data (torch.Tensor or numpy.ndarray): The image data.
            require_shape (bool): Whether to check that the shape of the data is the same as the current data.

        Raises:
            AssertionError: If `require_shape` is `True` and the shape of the data is different from the current data.
        """
        if self._data is not None and require_shape:
            assert data.shape == self._data.shape
        if data is None:
            self.data = torch.tensor(
                (), dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        elif isinstance(data, torch.Tensor):
            self._data = data.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        else:
            self._data = torch.as_tensor(
                data, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return self.__class__(
            data=torch.clone(self.data),
            header=self.header.copy(**kwargs),
            **kwargs,
        )

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is now filled with zeros.

        """
        return self.__class__(
            data=torch.zeros_like(self.data),
            header=self.header.copy(**kwargs),
            **kwargs,
        )

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        return self.__class__(
            data=self.data[window.get_indices(self)],
            header=self.header.get_window(window, **kwargs),
            **kwargs,
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        if self._data is not None:
            self._data = self._data.to(dtype=dtype, device=device)
        self.header.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):
        # does this show up?
        if len(pixels) == 1:  # same crop in all dimension
            self.set_data(
                self.data[
                    pixels[0].int() : (self.data.shape[0] - pixels[0]).int(),
                    pixels[0].int() : (self.data.shape[1] - pixels[0]).int(),
                ],
                require_shape=False,
            )
        elif len(pixels) == 2:  # different crop in each dimension
            self.set_data(
                self.data[
                    pixels[1].int() : (self.data.shape[0] - pixels[1]).int(),
                    pixels[0].int() : (self.data.shape[1] - pixels[0]).int(),
                ],
                require_shape=False,
            )
        elif len(pixels) == 4:  # different crop on all sides
            self.set_data(
                self.data[
                    pixels[2].int() : (self.data.shape[0] - pixels[3]).int(),
                    pixels[0].int() : (self.data.shape[1] - pixels[1]).int(),
                ],
                require_shape=False,
            )
        self.header = self.header.crop(pixels)
        return self

    def flatten(self, attribute: str = "data") -> np.ndarray:
        return getattr(self, attribute).reshape(-1)

    def get_coordinate_meshgrid(self):
        return self.header.get_coordinate_meshgrid()

    def get_coordinate_corner_meshgrid(self):
        return self.header.get_coordinate_corner_meshgrid()

    def get_coordinate_simps_meshgrid(self):
        return self.header.get_coordinate_simps_meshgrid()

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

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        return self.__class__(
            data=self.data[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .sum(axis=(1, 3)),
            header=self.header.reduce(scale, **kwargs),
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
        self.data = pad(self.data, pad=pad_boundaries, mode="constant", value=0)
        self.header.expand(padding)

    def _save_image_list(self):
        img_header = self.header._save_image_list()
        image_list = [
            fits.PrimaryHDU(self._data.detach().cpu().numpy(), header=img_header)
        ]
        return image_list

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
                self.set_data(np.array(hdu.data, dtype=np.float64), require_shape=False)
                break
        self.header.load(filename)
        return hdul

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data -= other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data += other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data += other
            return new_img

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data -= other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data += other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data += other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image):
            self.data[other.window.get_indices(self)] += other.data[
                self.window.get_indices(other)
            ]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            self.data[other.window.get_indices(self)] -= other.data[
                self.window.get_indices(other)
            ]
        else:
            self.data -= other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], Image):
            return self.get_window(args[0].window)
        raise ValueError("Unrecognized Image getitem request!")

    def __str__(self):
        return f"image pixelscale: {self.pixelscale} origin: {self.origin}\ndata: {self.data}"


class Image_List(Image):
    def __init__(self, image_list):
        self.image_list = list(image_list)

    @property
    def window(self):
        return Window_List(list(image.window for image in self.image_list))

    @property
    def pixelscale(self):
        return tuple(image.pixelscale for image in self.image_list)

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

    def get_window(self, window):
        return self.__class__(
            tuple(image[win] for image, win in zip(self.image_list, window)),
        )

    def index(self, other):
        if isinstance(other, Image) and hasattr(other, "identity"):
            for i, self_image in enumerate(self.image_list):
                if other.identity == self_image.identity:
                    return i
            else:
                raise ValueError(
                    "Could not find identity match between image list and input image"
                )
        raise NotImplementedError(f"Image_List cannot get index for {type(other)}")

    def to(self, dtype=None, device=None):
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        for image in self.image_list:
            image.to(dtype=dtype, device=device)
        return self

    def crop(self, *pixels):
        raise NotImplementedError("Crop function not available for Image_List object")

    def get_coordinate_meshgrid(self):
        return tuple(image.get_coordinate_meshgrid() for image in self.image_list)

    def get_coordinate_corner_meshgrid(self):
        return tuple(
            image.get_coordinate_corner_meshgrid() for image in self.image_list
        )

    def get_coordinate_simps_meshgrid(self):
        return tuple(image.get_coordinate_simps_meshgrid() for image in self.image_list)

    def flatten(self, attribute="data"):
        return torch.cat(tuple(image.flatten(attribute) for image in self.image_list))

    def reduce(self, scale):
        assert isinstance(scale, int) or scale.dtype is torch.int32
        if scale == 1:
            return self

        return self.__class__(
            tuple(image.reduce(scale) for image in self.image_list),
        )

    def __sub__(self, other):
        if isinstance(other, Image_List):
            new_list = []
            for self_image, other_image in zip(self.image_list, other.image_list):
                new_list.append(self_image - other_image)
            return self.__class__(new_list)
        else:
            new_list = []
            for self_image, other_image in zip(self.image_list, other):
                new_list.append(self_image - other_image)
            return self.__class__(new_list)

    def __add__(self, other):
        if isinstance(other, Image_List):
            new_list = []
            for self_image, other_image in zip(self.image_list, other.image_list):
                new_list.append(self_image + other_image)
            return self.__class__(new_list)
        else:
            new_list = []
            for self_image, other_image in zip(self.image_list, other):
                new_list.append(self_image + other_image)
            return self.__class__(new_list)

    def __isub__(self, other):
        if isinstance(other, Image_List):
            for self_image, other_image in zip(self.image_list, other.image_list):
                self_image -= other_image
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image -= other_image
        return self

    def __iadd__(self, other):
        if isinstance(other, Image_List):
            for self_image, other_image in zip(self.image_list, other.image_list):
                self_image += other_image
        else:
            for self_image, other_image in zip(self.image_list, other):
                self_image += other_image
        return self

    def save(self, filename=None, overwrite=True):
        raise NotImplementedError("Save/load not yet available for image lists")

    def load(self, filename):
        raise NotImplementedError("Save/load not yet available for image lists")

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], Image):
            return self.get_window(args[0].window)
        if all(isinstance(arg, (int, slice)) for arg in args):
            return self.image_list.__getitem__(*args)
        raise ValueError("Unrecognized Image_List getitem request!")

    def __str__(self):
        return f"image list of:\n" + "\n".join(
            image.__str__() for image in self.image_list
        )

    def __iter__(self):
        return (img for img in self.image_list)

    #     self._index = 0
    #     return self

    # def __next__(self):
    #     if self._index >= len(self.image_list):
    #         raise StopIteration
    #     img = self.image_list[self._index]
    #     self._index += 1
    #     return img
