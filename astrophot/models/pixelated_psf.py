import torch

from .psf_model_object import PSFModel
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from caskade import OverrideParam
from ..param import forward
from ..backend_obj import backend, ArrayLike

__all__ = ["PixelatedPSF"]


@combine_docstrings
class PixelatedPSF(PSFModel):
    """point source model which uses an image of the PSF as its
    representation for point sources. Using bilinear interpolation it
    will shift the PSF within a pixel to accurately represent the
    center location of a point source. There is no functional form for
    this object type as any image can be supplied. The image pixels
    will be optimized as individual parameters. This can very quickly
    result in a large number of parameters and a near impossible
    fitting task, ideally this should be restricted to a very small
    area likely at the center of the PSF.

    To initialize the PSF image will by default be set to the target
    PSF_Image values, thus one can use an empirical PSF as a starting
    point. Since only bilinear interpolation is performed, it is
    recommended to provide the PSF at a higher resolution than the
    image if it is near the nyquist sampling limit. Bilinear
    interpolation is very fast and accurate for smooth models, so this
    way it is possible to do the expensive interpolation before
    optimization and save time. Note that if you do this you must
    provide the PSF as a PSF_Image object with the correct pixelscale
    (essentially just divide the pixelscale by the upsampling factor
    you used).

    **Parameters:**
    -    `pixels`: the total flux within each pixel, represented as the log of the flux.

    """

    _model_type = "pixelated"
    _parameter_specs = {"pixels": {"units": "flux/arcsec^2"}}
    usable = True
    sampling_mode = "midpoint"
    integrate_mode = "none"

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        if self.pixels.initialized:
            return
        target_area = self.target[self.window]
        self.pixels.dynamic_value = backend.copy(target_area.data) / target_area.pixel_area

    @forward
    def brightness(
        self, x: ArrayLike, y: ArrayLike, pixels: ArrayLike, center: ArrayLike
    ) -> ArrayLike:
        with OverrideParam(self.target.crtan, center):
            i, j = self.target.plane_to_pixel(x, y)
        result = interp2d(pixels, i, j)
        return result
