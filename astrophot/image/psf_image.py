from typing import List, Optional, Union
import numpy as np

from ..errors import SpecificationConflict, InvalidImage

from .jacobian_image import JacobianImage
from .. import config
from ..backend_obj import backend, ArrayLike
from .mixins import DataMixin
from .window import Window
from . import func
from ..utils.decorators import combine_docstrings

__all__ = ("PSFImage",)


@combine_docstrings
class PSFImage(DataMixin):
    """Image object which represents a model of PSF (Point Spread Function).

    PSFImage inherits from the base Image class and represents the model of a point spread function.
    The point spread function characterizes the response of an imaging system to a point source or point object.

    The shape of the PSF data should be odd (for your sanity) but this is not enforced.
    """

    base_scale = 1.0

    def __init__(
        self,
        *,
        data: Optional[ArrayLike] = None,
        upsample: int = 1,
        crpix: Optional[tuple[int, int]] = None,
        filename: Optional[str] = None,
        hduext: int = 0,
        identity: str = None,
        _data: Optional[ArrayLike] = None,
        **kwargs,
    ):
        if _data is None:
            self.data = data
        else:
            self._data = _data

        super().__init__(**kwargs)
        self.upsample = upsample
        self.crpix = crpix

        if identity is None:
            self._identity = id(self)
        else:
            self._identity = identity

        if filename is not None:
            self.load(filename, hduext=hduext)
            return

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        norm = backend.sum(self._data, dim=(-2, -1), keepdim=True)
        self._data = self._data / norm
        self._weight = self._weight * norm**2

    @property
    def identity(self):
        return self._identity

    @property
    def window(self) -> Window:
        return Window(window=((0, 0), self._data.shape[:2]), image=self)

    @property
    def data(self):
        """The image data, which is a Array of pixel values."""
        return backend.transpose(self._data, 1, 0)

    @data.setter
    def data(self, value: Optional[ArrayLike]):
        """Set the image data. If value is None, the data is initialized to an empty Array."""
        if value is None:
            self._data = backend.ones((1, 1), dtype=config.DTYPE, device=config.DEVICE)
        else:
            assert all(
                s % 2 == 1 for s in value.shape[-2:]
            ), "PSF data shape must be odd in both dimensions."
            # Transpose since pytorch uses (j, i) indexing when (i, j) is more natural for coordinates
            self._data = backend.transpose(
                backend.as_array(value, dtype=config.DTYPE, device=config.DEVICE), 1, 0
            )

    @property
    def upsample(self) -> int:
        return self._upsample

    @upsample.setter
    def upsample(self, value: int):
        if value < 1:
            raise ValueError("upsample factor must be a positive integer.")
        self._upsample = int(value)

    @property
    def pad(self) -> int:
        return max(self._data.shape[-2:]) // 2

    @property
    def crpix(self):
        if self._crpix is None:
            return np.array([self._data.shape[-2] // 2, self._data.shape[-1] // 2])
        return self._crpix

    @crpix.setter
    def crpix(self, value):
        if value is None:
            self._crpix = None
        else:
            self._crpix = np.array(value)

    @property
    def pixel_area(self):
        return backend.as_array(1.0 / self.upsample**2, dtype=config.DTYPE, device=config.DEVICE)

    @property
    def pixelscale(self):
        return backend.as_array(1.0 / self.upsample, dtype=config.DTYPE, device=config.DEVICE)

    @property
    def flip_ra_axis(self):
        return False

    @property
    def zeropoint(self):
        return None

    def flatten(self, attribute: str = "data") -> ArrayLike:
        return backend.flatten(getattr(self, attribute), end_dim=1)

    def pixel_collecting_area(self, *args, **kwargs):
        return 1.0

    def targpixel_to_mypixel(self, I_, J_):
        """
        Convert between coordinate spaces. "targpixel" refers to the pixel
        coordinates of the target of this PSF, which have the origin at the
        center of the PSF and a step of 1 corresponds to one target pixel
        length. "mypixel" refers to the pixel coordinates of this PSF image,
        which have an origin at the center of the [0,0] pixel and a step of 1
        corresponds to one PSF pixel length (which is 1/upsample of a target
        pixel length).
        """
        return I_ * self.upsample + self.crpix[0], J_ * self.upsample + self.crpix[1]

    def mypixel_to_targpixel(self, i, j):
        """
        Convert between coordinate spaces. "targpixel" refers to the pixel
        coordinates of the target of this PSF, which have the origin at the
        center of the PSF and a step of 1 corresponds to one target pixel
        length. "mypixel" refers to the pixel coordinates of this PSF image,
        which have an origin at the center of the [0,0] pixel and a step of 1
        corresponds to one PSF pixel length (which is 1/upsample of a target
        pixel length).
        """
        return (i - self.crpix[0]) / self.upsample, (j - self.crpix[1]) / self.upsample

    def pixel_center_meshgrid(self, window=None, pad=0, upsample=1) -> tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, centered on the pixel grid."""
        if window is None:
            window = self.window
        return func.pixel_center_meshgrid(window.extent, pad, upsample, config.DTYPE, config.DEVICE)

    def pixel_corner_meshgrid(self, window=None, pad=0, upsample=1) -> tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with corners at the pixel grid."""
        if window is None:
            window = self.window
        return func.pixel_corner_meshgrid(window.extent, pad, upsample, config.DTYPE, config.DEVICE)

    def pixel_simpsons_meshgrid(
        self, window=None, pad=0, upsample=1
    ) -> tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with Simpson's rule sampling."""
        if window is None:
            window = self.window
        return func.pixel_simpsons_meshgrid(
            window.extent, pad, upsample, config.DTYPE, config.DEVICE
        )

    def pixel_quad_meshgrid(
        self, window=None, pad=0, upsample=1, order=3
    ) -> tuple[ArrayLike, ArrayLike]:
        """Get a meshgrid of pixel coordinates in the image, with quadrature sampling."""
        if window is None:
            window = self.window
        return func.pixel_quad_meshgrid(
            window.extent, pad, upsample, config.DTYPE, config.DEVICE, order=order
        )

    def coordinate_center_meshgrid(self) -> tuple[ArrayLike, ArrayLike]:
        i, j = self.pixel_center_meshgrid()
        return self.mypixel_to_targpixel(i, j)

    def coordinate_corner_meshgrid(self) -> tuple[ArrayLike, ArrayLike]:
        i, j = self.pixel_corner_meshgrid()
        return self.mypixel_to_targpixel(i, j)

    def coordinate_simpsons_meshgrid(self) -> tuple[ArrayLike, ArrayLike]:
        i, j = self.pixel_simpsons_meshgrid()
        return self.mypixel_to_targpixel(i, j)

    def coordinate_quad_meshgrid(self, order=3) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        i, j, _ = self.pixel_quad_meshgrid(order=order)
        return self.mypixel_to_targpixel(i, j)

    def get_indices(self, other: Window):
        if other.image is self:
            return slice(max(0, other.i_low), min(self._data.shape[0], other.i_high)), slice(
                max(0, other.j_low), min(self._data.shape[1], other.j_high)
            )
        if other.image.identity == self.identity:
            shift = np.round(self.crpix - other.crpix).astype(int)
            return slice(
                min(max(0, other.i_low + shift[0]), self._data.shape[0]),
                max(0, min(other.i_high + shift[0], self._data.shape[0])),
            ), slice(
                min(max(0, other.j_low + shift[1]), self._data.shape[1]),
                max(0, min(other.j_high + shift[1], self._data.shape[1])),
            )
        raise RuntimeError(
            f"Cannot get indices for window with different image! Window image: {other.image.name}, self image: {self.name}"
        )

    def jacobian_image(
        self,
        parameters: Optional[List[str]] = None,
        data: Optional[ArrayLike] = None,
        **kwargs,
    ) -> JacobianImage:
        """
        Construct a blank `JacobianImage` object formatted like this current `PSFImage` object. Mostly used internally.
        """
        if parameters is None:
            data = None
            parameters = []
        elif data is None:
            data = backend.zeros(
                (*self._data.shape, len(parameters)),
                dtype=config.DTYPE,
                device=config.DEVICE,
            )
        kwargs = {
            "CD": ((1, 0), (0, 1)),
            "crpix": self.crpix,
            "crtan": (0, 0),
            "crval": (0, 0),
            "identity": self.identity,
            "_data": data,
            **kwargs,
        }
        return JacobianImage(parameters=parameters, **kwargs)

    def model_image(self, window=None, **kwargs) -> "PSFImage":
        """
        Construct a blank `ModelImage` object formatted like this current `TargetImage` object. Mostly used internally.
        """
        if window is None:
            window = self.window
        si, sj = self.get_indices(window)
        kwargs = {
            "_data": backend.zeros_like(self._data[si, sj]),
            "crpix": self.crpix - np.array((si.start, sj.start)),
            "upsample": self.upsample,
            "identity": self.identity,
            **kwargs,
        }
        return PSFImage(**kwargs)

    def fits_info(self) -> dict:
        return {
            "UPSMPL": self.upsample,
            "CRPIX1": self.crpix[0],
            "CRPIX2": self.crpix[1],
            "IDNTY": self.identity,
        }

    def __iadd__(self, other: Union["PSFImage", ArrayLike]):
        if isinstance(other, (int, float, backend.array_type)):
            self._data = self._data + other
            return self
        if not isinstance(other, PSFImage):
            raise InvalidImage(f"PSF images can only add with each other, not: {type(other)}")

        if self.upsample != other.upsample:
            raise SpecificationConflict("Cannot add PSF images with different upsample factors.")

        islice = slice(
            max(0, self.crpix[0] - other.crpix[0]),
            min(self._data.shape[-2], self.crpix[0] + other._data.shape[-2] - other.crpix[0]),
        )
        jslice = slice(
            max(0, self.crpix[1] - other.crpix[1]),
            min(self._data.shape[-1], self.crpix[1] + other._data.shape[-1] - other.crpix[1]),
        )
        self._data = backend.add_at_indices(self._data, (islice, jslice), other._data)
        return self

    def __sub__(self, other: "PSFImage"):
        return self.copy(_data=self._data - other._data)

    def copy_kwargs(self, **kwargs) -> dict:
        kwargs = {
            "_data": backend.copy(self._data),
            "crpix": self.crpix,
            "upsample": self.upsample,
            "identity": self.identity,
            **kwargs,
        }
        return super().copy_kwargs(**kwargs)

    def copy(self, **kwargs):
        """Produce a copy of this image with all of the same properties. This
        can be used when one wishes to make temporary modifications to
        an image and then will want the original again.

        """
        return self.__class__(**self.copy_kwargs(**kwargs))

    def reduce(self, scale: int):
        return self.copy(upsample=self.upsample * scale)

    def get_window(self, other: Union[Window, "Image"], indices=None, **kwargs):
        """Get a new image object which is a window of this image
        corresponding to the other image's window. This will return a
        new image object with the same properties as this one, but with
        the data cropped to the other image's window.

        """
        if indices is None:
            indices = self.get_indices(other if isinstance(other, Window) else other.window)
        new_img = self.copy(
            _data=self._data[indices],
            crpix=self.crpix - np.array((indices[0].start, indices[1].start)),
            **kwargs,
        )
        return new_img

    def __getitem__(self, *args):
        if len(args) == 1 and isinstance(args[0], Window):
            return self.get_window(args[0])
        return super().__getitem__(*args)
