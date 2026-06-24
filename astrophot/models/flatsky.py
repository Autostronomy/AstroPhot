import numpy as np

from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from .sky_model_object import SkyModel
from ..backend_obj import backend, ArrayLike
from ..param import forward

__all__ = ["FlatSky"]


@combine_docstrings
class FlatSky(SkyModel):
    """Model for the sky background in which all values across the image
    are the same.

    :param I0: brightness for the sky in flux/arcsec^2

    """

    _model_type = "flat"
    _parameter_specs = {
        "I0": {
            "units": "flux/arcsec^2",
            "shape": (),
            "dynamic": True,
            "description": "brightness for the sky in flux/arcsec^2",
        }
    }
    usable = True

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.I0.initialized:
            return

        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area._data).copy()
        mask = backend.to_numpy(target_area._mask)
        dat[mask] = np.median(dat[~mask])

        self.I0.value = np.median(dat) / self.target.pixel_area.item()

    @forward
    def brightness(self, x: ArrayLike, y: ArrayLike, I0: ArrayLike) -> ArrayLike:
        return backend.ones_like(x) * I0
