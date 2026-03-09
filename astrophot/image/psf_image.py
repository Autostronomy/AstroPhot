from typing import List, Optional

import numpy as np

from ..param import Param
from .jacobian_image import JacobianImage
from .. import config
from ..backend_obj import backend, ArrayLike
from .mixins import DataMixin

__all__ = ("PSFImage",)


class PSFImage(DataMixin):
    """Image object which represents a model of PSF (Point Spread Function).

    PSFImage inherits from the base Image class and represents the model of a point spread function.
    The point spread function characterizes the response of an imaging system to a point source or point object.

    The shape of the PSF data should be odd (for your sanity) but this is not enforced.
    """

    def __init__(
        self,
        *args,
        upsample: int = 1,
        crpix: tuple[float, float] = (0.0, 0.0),
        filename: Optional[str] = None,
        hduext: int = 0,
        identity: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.upsample = upsample
        self.crpix = crpix

        if identity is None:
            self.identity = id(self)
        else:
            self.identity = identity

        if filename is not None:
            self.load(filename, hduext=hduext)
            return

    def normalize(self):
        """Normalizes the PSF image to have a sum of 1."""
        norm = backend.sum(self.value, dim=(-2, -1), keepdim=True)
        self.value = self.value / norm
        self._weight = self._weight * norm**2

    @Param.value.getter
    def value(self):
        value = super().value
        if value is None:
            value = backend.ones((1, 1), dtype=config.DTYPE, device=config.DEVICE)
        return value

    @property
    def upsample(self) -> int:
        if len(self.children) > 0:
            return next(iter(self.children.values)).upsample
        return self._upsample

    @upsample.setter
    def upsample(self, value: int):
        if value < 1:
            raise ValueError("upsample factor must be a positive integer.")
        self._upsample = int(value)

    @property
    def pixelscale(self) -> float:
        return 1.0 / self.upsample

    @property
    def psf_pad(self) -> int:
        return max(self.value.shape[-2:]) // 2

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
            "CD": self.CD.value,
            "crpix": self.crpix,
            "crtan": self.crtan.value,
            "crval": self.crval.value,
            "zeropoint": self.zeropoint,
            "identity": self.identity,
            **kwargs,
        }
        return JacobianImage(parameters=parameters, _data=data, **kwargs)

    def model_image(self, **kwargs) -> "PSFImage":
        """
        Construct a blank `ModelImage` object formatted like this current `TargetImage` object. Mostly used internally.
        """
        kwargs = {
            "_data": backend.zeros_like(self._data),
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
