from typing import Optional, Union, Any, Sequence, Tuple
from copy import deepcopy

import torch
from torch.nn.functional import pad
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from .window_object import Window, Window_List
from .image_header import Image_Header
from .. import AP_config
from ..errors import SpecificationConflict, ConflicingWCS, InvalidData, InvalidWindow

__all__ = ["Image", "Image_List"]


class Image(object):
    """Core class to represent images with pixel values, pixel scale,
       and a window defining the spatial coordinates on the sky.
       It supports arithmetic operations with other image objects while preserving logical image boundaries.
       It also provides methods for determining the coordinate locations of pixels

    Parameters:
        data: the matrix of pixel values for the image
        pixelscale: the length of one side of a pixel in arcsec/pixel
        window: an AstroPhot Window object which defines the spatial cooridnates on the sky
        filename: a filename from which to load the image.
        zeropoint: photometric zero point for converting from pixel flux to magnitude
        metadata: Any information the user wishes to associate with this image, stored in a python dictionary
        origin: The origin of the image in the coordinate system.
    """

    def __init__(
        self,
        *,
        data: Optional[torch.Tensor] = None,
        header: Optional[Image_Header] = None,
        wcs: Optional[AstropyWCS] = None,
        pixelscale: Optional[Union[float, torch.Tensor]] = None,
        window: Optional[Window] = None,
        filename: Optional[str] = None,
        zeropoint: Optional[Union[float, torch.Tensor]] = None,
        metadata: Optional[dict] = None,
        origin: Optional[Sequence] = None,
        center: Optional[Sequence] = None,
        identity: str = None,
        state: Optional[dict] = None,
        fits_state: Optional[dict] = None,
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
        window : Window or None, optional
            A Window object defining the area of the image to use. Default is None.
        filename : str or None, optional
            The name of a file containing the image data. Default is None.
        zeropoint : float or None, optional
            The image's zeropoint, used for flux calibration. Default is None.
        metadata : dict or None, optional
            Any information the user wishes to associate with this image, stored in a python dictionary. Default is None.
        origin : numpy.ndarray or None, optional
            The origin of the image in the coordinate system, as a 1D array of length 2. Default is None.
        center : numpy.ndarray or None, optional
            The center of the image in the coordinate system, as a 1D array of length 2. Default is None.

        Returns:
        --------
        None
        """
        self._data = None

        if state is not None:
            self.header = Image_Header(state=state["header"])
        elif fits_state is not None:
            self.set_fits_state(fits_state)
            return
        elif header is None:
            if data is None and window is None and filename is None:
                raise InvalidData("Image must have either data or a window to construct itself.")
            self.header = Image_Header(
                data_shape=None if data is None else data.shape,
                pixelscale=pixelscale,
                wcs=wcs,
                window=window,
                filename=filename,
                zeropoint=zeropoint,
                metadata=metadata,
                origin=origin,
                center=center,
                identity=identity,
                **kwargs,
            )
        else:
            self.header = header

        if filename is not None:
            self.load(filename)
        elif state is not None:
            self.set_state(state)
        elif fits_state is not None:
            self.data = fits_state[0]["DATA"]
        else:
            # set the data
            if data is None:
                self.data = torch.zeros(
                    torch.flip(self.window.pixel_shape, (0,)).detach().cpu().tolist(),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
            else:
                self.data = data

            self.to()

        # # Check that image data and header are in agreement (this requires talk back from GPU to CPU so is only used for testing)
        # assert np.all(np.flip(np.array(self.data.shape)[:2]) == self.window.pixel_shape.numpy()), f"data shape {np.flip(np.array(self.data.shape)[:2])}, window shape {self.window.pixel_shape.numpy()}"

    @property
    def north(self):
        return self.header.north

    @property
    def pixel_area(self):
        return self.header.pixel_area

    @property
    def pixel_length(self):
        return self.header.pixel_length

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
    def size(self) -> torch.Tensor:
        """
        Returns the size of the image window, the number of pixels in the image.

        Returns:
            torch.Tensor: A 0D tensor containing the number of pixels.
        """
        return self.header.window.size

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
    def metadata(self):
        return self.header.metadata

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

    def set_data(self, data: Union[torch.Tensor, np.ndarray], require_shape: bool = True):
        """
        Set the image data.

        Args:
            data (torch.Tensor or numpy.ndarray): The image data.
            require_shape (bool): Whether to check that the shape of the data is the same as the current data.

        Raises:
            SpecificationConflict: If `require_shape` is `True` and the shape of the data is different from the current data.
        """
        if self._data is not None and require_shape and data.shape != self._data.shape:
            raise SpecificationConflict(
                f"Attempting to change image data with tensor that has a different shape! ({data.shape} vs {self._data.shape}) Use 'require_shape = False' if this is desired behaviour."
            )

        if data is None:
            self.data = torch.tensor((), dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        elif isinstance(data, torch.Tensor):
            self._data = data.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        else:
            self._data = torch.as_tensor(data, dtype=AP_config.ap_dtype, device=AP_config.ap_device)

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
            data=self.data[self.window.get_self_indices(window)],
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
        if not isinstance(scale, int) and not (
            isinstance(scale, torch.Tensor) and scale.dtype is torch.int32
        ):
            raise SpecificationConflict(f"Reduce scale must be an integer! not {type(scale)}")
        if scale == 1:
            return self

        MS = self.data.shape[0] // scale
        NS = self.data.shape[1] // scale
        return self.__class__(
            data=self.data[: MS * scale, : NS * scale]
            .reshape(MS, scale, NS, scale)
            .sum(axis=(1, 3)),
            header=self.header.rescale_pixel(scale, **kwargs),
            **kwargs,
        )

    def expand(self, padding: Tuple[float]) -> None:
        """
        Args:
          padding tuple[float]: length 4 tuple with amounts to pad each dimension in physical units
        """
        padding = np.array(padding)
        if np.any(padding < 0):
            raise SpecificationConflict("negative padding not allowed in expand method")
        pad_boundaries = tuple(np.int64(np.round(np.array(padding) / self.pixelscale)))
        self.data = pad(self.data, pad=pad_boundaries, mode="constant", value=0)
        self.header.expand(padding)

    def get_state(self):
        state = {}
        state["type"] = self.__class__.__name__
        state["data"] = self.data.detach().cpu().tolist()
        state["header"] = self.header.get_state()
        return state

    def set_state(self, state):
        self.set_data(state["data"], require_shape=False)
        self.header.set_state(state["header"])

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

    def __sub__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data -= other.data[self.window.get_other_indices(other)]
            return new_img
        else:
            new_img = self.copy()
            new_img.data -= other
            return new_img

    def __add__(self, other):
        if isinstance(other, Image):
            new_img = self[other.window].copy()
            new_img.data += other.data[self.window.get_other_indices(other)]
            return new_img
        else:
            new_img = self.copy()
            new_img.data += other
            return new_img

    def __iadd__(self, other):
        if isinstance(other, Image):
            self.data[other.window.get_other_indices(self)] += other.data[
                self.window.get_other_indices(other)
            ]
        else:
            self.data += other
        return self

    def __isub__(self, other):
        if isinstance(other, Image):
            self.data[other.window.get_other_indices(self)] -= other.data[
                self.window.get_other_indices(other)
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
        return f"image pixelscale: {self.pixelscale.detach().cpu().numpy()} origin: {self.origin.detach().cpu().numpy()} shape: {self.shape.detach().cpu().numpy()}"

    def __repr__(self):
        return f"image pixelscale: {self.pixelscale.detach().cpu().numpy()} origin: {self.origin.detach().cpu().numpy()} shape: {self.shape.detach().cpu().numpy()} center: {self.center.detach().cpu().numpy()}\ndata: {self.data.detach().cpu().numpy()}"


class Image_List(Image):
    def __init__(self, image_list, window=None):
        self.image_list = list(image_list)
        self.check_wcs()
        self.window = window

    def check_wcs(self):
        """Ensure the WCS systems being used by all the windows in this list
        are consistent with each other. They should all project world
        coordinates onto the same tangent plane.

        """
        ref = torch.stack(tuple(I.window.reference_radec for I in self.image_list))
        if not torch.allclose(ref, ref[0]):
            raise ConflicingWCS(
                "Reference (world) coordinate mismatch! All images in Image_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )
        ref = torch.stack(tuple(I.window.reference_planexy for I in self.image_list))
        if not torch.allclose(ref, ref[0]):
            raise ConflicingWCS(
                "Reference (tangent plane) coordinate mismatch! All images in Image_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )

        if len(set(I.window.projection for I in self.image_list)) > 1:
            raise ConflicingWCS(
                "Projection mismatch! All images in Image_List are not on the same tangent plane! Likely serious coordinate mismatch problems. See the coordinates page in the documentation for what this means."
            )

    @property
    def window(self):
        return Window_List(list(image.window for image in self.image_list))

    @window.setter
    def window(self, window):
        if window is None:
            return

        if not isinstance(window, Window_List):
            raise InvalidWindow("Target_List must take a Window_List object as its window")

        for i in range(len(self.image_list)):
            self.image_list[i] = self.image_list[i][window.window_list[i]]

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
                raise ValueError("Could not find identity match between image list and input image")
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
        return tuple(image.get_coordinate_corner_meshgrid() for image in self.image_list)

    def get_coordinate_simps_meshgrid(self):
        return tuple(image.get_coordinate_simps_meshgrid() for image in self.image_list)

    def flatten(self, attribute="data"):
        return torch.cat(tuple(image.flatten(attribute) for image in self.image_list))

    def reduce(self, scale):
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
        return f"image list of:\n" + "\n".join(image.__str__() for image in self.image_list)

    def __repr__(self):
        return f"image list of:\n" + "\n".join(image.__repr__() for image in self.image_list)

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
