import numpy as np

from .base import Model
from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from ..image import PSFImage
from ..errors import SpecificationConflict
from ..param import forward
from ..backend_obj import backend, ArrayLike
from .. import config
from . import func

__all__ = ("PointSource",)


@combine_docstrings
class PointSource(ComponentModel):
    """Describes a point source in the image, this is a delta function at
    some position in the sky. This is typically used to describe stars,
    supernovae, quasars, asteroids or any other object which can essentially be
    entirely described by a position and total flux (no structure).

    :param flux: The total flux of the point source

    """

    _model_type = "point"
    _parameter_specs = {
        "flux": {
            "units": "flux",
            "valid": (0, None),
            "shape": (),
            "dynamic": True,
            "description": "The total flux of the point source",
        },
    }
    internal_psf = True
    usable = True

    def __init__(self, *args, integrate_mode="none", **kwargs):
        super().__init__(*args, integrate_mode=integrate_mode, **kwargs)

    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.psf is None:
            raise SpecificationConflict("PointSource needs a psf!")

        if self.flux.initialized:
            return
        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area._data).copy()
        mask = backend.to_numpy(target_area._mask)
        dat[mask] = np.median(dat[~mask])

        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.median(edge)
        self.flux.value = np.abs(np.sum(dat - edge_average))

    @property
    def integrate_mode(self):
        return "none"

    @integrate_mode.setter
    def integrate_mode(self, value):
        if value != "none":
            config.logger.warning(
                "PointSource models are restricted to integrate mode of 'none', ignoring integrate_mode setting."
            )

    # Psf convolution should be on by default since this is a delta function
    @property
    def psf_convolve(self):
        return True

    @psf_convolve.setter
    def psf_convolve(self, value):
        pass

    def _prep_psf(self):
        psf = self.psf

        if isinstance(psf, PSFImage):
            return psf._data, psf.upsample, 0
        if isinstance(psf, Model):
            return psf, psf.upsample, 0
        return None, 1, 0

    @forward
    def sample(
        self,
        I_: ArrayLike,
        J_: ArrayLike,
        psf: ArrayLike = None,
        crop: int = 0,
        downsample: int = 1,
        center=None,
        flux=None,
        _CD=None,
        _crtan=None,
        _crpix=None,
    ):
        if isinstance(psf, Model):
            psf = psf()._data
        if _CD is None:
            i0, j0 = self.target.plane_to_pixel(*center)
        else:
            i0, j0 = self.target.plane_to_pixel(*center, CD=_CD, crtan=_crtan, _crpix=_crpix)
        Z = interp2d(
            psf,
            (I_ - i0) * downsample + (psf.shape[0] // 2),
            (J_ - j0) * downsample + (psf.shape[1] // 2),
        )
        Z = self._pixel_integrator(Z)
        Z = Z * flux
        Z = func.downsample(Z, downsample)
        return Z
