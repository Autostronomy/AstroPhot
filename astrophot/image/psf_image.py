from typing import List, Optional

import numpy as np

from .image_object import Image
from .jacobian_image import JacobianImage
from .. import config
from ..backend_obj import backend, ArrayLike
from .mixins import DataMixin

__all__ = ["PSFImage"]


class PSFImage(DataMixin, Image):
    """Image object which represents a model of PSF (Point Spread Function).

    PSFImage inherits from the base Image class and represents the model of a point spread function.
    The point spread function characterizes the response of an imaging system to a point source or point object.

    The shape of the PSF data should be odd (for your sanity) but this is not enforced.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"crval": (0, 0), "crpix": (0, 0), "crtan": (0, 0)})
        super().__init__(*args, **kwargs)
        self.crpix = (np.array(self.data.shape, dtype=np.float64) - 1.0) / 2

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        norm = backend.sum(self.data)
        self._data = self.data / norm
        if self.has_weight:
            self._weight = self.weight * norm**2

    @property
    def psf_pad(self) -> int:
        return max(self.data.shape) // 2

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
                (*self.data.shape, len(parameters)),
                dtype=config.DTYPE,
                device=config.DEVICE,
            )
        kwargs = {
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            **kwargs,
        }
        return JacobianImage(parameters=parameters, data=data, **kwargs)

    def model_image(self, **kwargs) -> "PSFImage":
        """
        Construct a blank `ModelImage` object formatted like this current `TargetImage` object. Mostly used internally.
        """
        kwargs = {
            "data": backend.zeros_like(self.data),
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "identity": self.identity,
            **kwargs,
        }
        return PSFImage(**kwargs)

    @property
    def zeropoint(self):
        return None

    @zeropoint.setter
    def zeropoint(self, value):
        """PSFImage does not support zeropoint."""
        pass

    def plane_to_world(self, x, y):
        raise NotImplementedError(
            "PSFImage does not support plane_to_world conversion. There is no meaningful world position of a PSF image."
        )

    def world_to_plane(self, ra, dec):
        raise NotImplementedError(
            "PSFImage does not support world_to_plane conversion. There is no meaningful world position of a PSF image."
        )
