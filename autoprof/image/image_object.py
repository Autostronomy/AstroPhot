from typing import Optional, Union, Any, Sequence, Tuple
from copy import deepcopy

import torch
from torch.nn.functional import pad
import numpy as np
from astropy.io import fits

from .window_object import Window, Window_List
from .. import AP_config

__all__ = ["BaseImage", "Image_List"]


class BaseImage(object):
    """Core class to represent images with pixel values, pixel scale,
       and a window defining the spatial coordinates on the sky.
       It supports arithmetic operations with other image objects while preserving logical image boundaries.
       It also provides methods for determining the coordinate locations of pixels

    Parameters:
        data: the matrix of pixel values for the image
        pixelscale: the length of one side of a pixel in arcsec/pixel
        window: an AutoProf Window object which defines the spatial cooridnates on the sky
        filename: a filename from which to load the image.
        zeropoint: photometric zero point for converting from pixel flux to magnitude
        note: a note about this image if any
        origin: The origin of the image in the coordinate system.
    """

    def __init__(
        self,
        data: Optional[Union[torch.Tensor]] = None,
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
        data : numpy.ndarray or None, optional
            The image data. Default is None.
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

        # Record identity
        if _identity is None:
            self.identity = str(id(self))
        else:
            self.identity = _identity

        if filename is not None:
            self.load(filename)
            return

        assert not (pixelscale is None and window is None)

        # set the data
        self.data = data

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
                        data.shape, dtype=AP_config.ap_dtype, device=AP_config.ap_device
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
                self.pixelscale = self.window.shape[0] / self.data.shape[1]
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
            zeropoint=self.zeropoint,
            origin=self.origin,
            note=self.note,
            window=self.window,
            _identity=self.identity,
            **kwargs,
        )

    def blank_copy(self, **kwargs):
        """Produces a blank copy of the image which has the same properties
        except that its data is not filled with zeros.

        """
        return self.__class__(
            data=torch.zeros_like(self.data),
            zeropoint=self.zeropoint,
            origin=self.origin,
            note=self.note,
            window=self.window,
            _identity=self.identity,
            **kwargs,
        )

    def get_window(self, window, **kwargs):
        """Get a sub-region of the image as defined by a window on the sky."""
        return self.__class__(
            data=self.data[window.get_indices(self)],
            pixelscale=self.pixelscale,
            zeropoint=self.zeropoint,
            note=self.note,
            origin=(self.window & window).origin,
            _identity=self.identity,
            **kwargs,
        )

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        if self._data is not None:
            self._data = self._data.to(dtype=dtype, device=device)
        self.window.to(dtype=dtype, device=device)
        return self

    def crop(self, pixels):
        # does this show up?
        if len(pixels) == 1:  # same crop in all dimension
            self.set_data(
                self.data[
                    pixels[0] : self.data.shape[0] - pixels[0],
                    pixels[0] : self.data.shape[1] - pixels[0],
                ],
                require_shape=False,
            )
            self.window -= pixels[0] * self.pixelscale
        elif len(pixels) == 2:  # different crop in each dimension
            self.set_data(
                self.data[
                    pixels[1] : self.data.shape[0] - pixels[1],
                    pixels[0] : self.data.shape[1] - pixels[0],
                ],
                require_shape=False,
            )
            self.window -= (
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                * self.pixelscale
            )
        elif len(pixels) == 4:  # different crop on all sides
            self.set_data(
                self.data[
                    pixels[2] : self.data.shape[0] - pixels[3],
                    pixels[0] : self.data.shape[1] - pixels[1],
                ],
                require_shape=False,
            )
            self.window -= (
                torch.as_tensor(
                    pixels, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
                * self.pixelscale
            )
        return self

    def flatten(self, attribute: str = "data") -> np.ndarray:
        return getattr(self, attribute).reshape(-1)

    def get_coordinate_meshgrid_np(self, x: float = 0.0, y: float = 0.0) -> np.ndarray:
        return self.window.get_coordinate_meshgrid_np(self.pixelscale, x, y)

    def get_coordinate_meshgrid_torch(self, x=0.0, y=0.0):
        return self.window.get_coordinate_meshgrid_torch(self.pixelscale, x, y)

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
            pixelscale=self.pixelscale * scale,
            zeropoint=self.zeropoint,
            note=self.note,
            window=self.window.make_copy(),
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
        self.data = pad(self.data, pad=pad_boundaries, mode="constant", value=0)
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
                self.pixelscale = eval(hdu.header.get("PXLSCALE"))
                self.zeropoint = eval(hdu.header.get("ZEROPNT"))
                self.note = hdu.header.get("NOTE")
                self.window = Window(**eval(hdu.header.get("WINDOW")))
                break
        return hdul

    def __sub__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                raise IndexError("images have no overlap, cannot subtract!")
            new_img = self[other.window].copy()
            new_img.data -= other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            new_img = self[other.window].copy()
            new_img.data += other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data += other
            return new_img

    def __sub__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                raise IndexError("images have no overlap, cannot subtract!")
            new_img = self[other.window].copy()
            new_img.data -= other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            new_img = self[other.window].copy()
            new_img.data += other.data[self.window.get_indices(other)]
            return new_img
        else:
            new_img = self[other.window.get_indices(self)].copy()
            new_img.data += other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot add images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            self.data[other.window.get_indices(self)] += other.data[
                self.window.get_indices(other)
            ]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, BaseImage):
            if not torch.isclose(self.pixelscale, other.pixelscale):
                raise IndexError("Cannot subtract images with different pixelscale!")
            if torch.any(self.origin + self.shape < other.origin) or torch.any(
                other.origin + other.shape < self.origin
            ):
                return self
            self.data[other.window.get_indices(self)] -= other.data[
                self.window.get_indices(other)
            ]
        else:
            self.data -= other
        return self

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Window):
            return self.get_window(args[0])
        if len(args) == 1 and isinstance(args[0], BaseImage):
            return self.get_window(args[0].window)
        raise ValueError("Unrecognized BaseImage getitem request!")

    def __str__(self):
        return f"image pixelscale: {self.pixelscale} origin: {self.origin}\ndata: {self.data}"


class Image_List(BaseImage):
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
        if isinstance(other, BaseImage) and hasattr(other, "identity"):
            for i, self_image in enumerate(self.image_list):
                if other.identity == self_image.identity:
                    return i
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

    def get_coordinate_meshgrid_np(self, x=0.0, y=0.0):
        return tuple(
            image.get_coordinate_meshgrid_np(x, y) for image in self.image_list
        )

    def get_coordinate_meshgrid_torch(self, x=0.0, y=0.0):
        return tuple(
            image.get_coordinate_meshgrid_torch(x, y) for image in self.image_list
        )

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
        if len(args) == 1 and isinstance(args[0], BaseImage):
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
