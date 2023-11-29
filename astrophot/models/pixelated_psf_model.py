import torch

from .psf_model_object import PSF_Model
from ..image import PSF_Image
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import interp2d
from ._shared_methods import select_target
from ..param import Param_Unlock, Param_SoftLimits
from .. import AP_config

__all__ = ["Pixelated_PSF"]


class Pixelated_PSF(PSF_Model):
    """point source model which uses an image of the PSF as its
    representation for point sources. Using bilinear interpolation it
    will shift the PSF within a pixel to accurately represent the
    center location of a point source. There is no funcitonal form for
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

    Parameters:
        pixels: the total flux within each pixel, represented as the log of the flux.

    """

    model_type = f"pixelated {PSF_Model.model_type}"
    parameter_specs = {
        "pixels": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = PSF_Model._parameter_order + ("pixels",)
    useable = True
    model_integrated = True

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)
        target_area = target[self.window]
        with Param_Unlock(parameters["pixels"]), Param_SoftLimits(parameters["pixels"]):
            if parameters["pixels"].value is None:
                dat = torch.abs(target_area.data)
                dat[dat == 0] = torch.median(dat) * 1e-7
                parameters["pixels"].value = torch.log10(dat / target.pixel_area)
            if parameters["pixels"].uncertainty is None:
                parameters["pixels"].uncertainty = torch.abs(parameters["pixels"].value) * self.default_uncertainty

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]

        # Convert coordinates into pixel locations in the psf image
        pX, pY = self.target.plane_to_pixel(X, Y)

        # Select only the pixels where the PSF image is defined
        select = torch.logical_and(
            torch.logical_and(pX > -0.5, pX < parameters["pixels"].shape[1] - 0.5),
            torch.logical_and(pY > -0.5, pY < parameters["pixels"].shape[0] - 0.5),
        )

        # Zero everywhere outside the psf
        result = torch.zeros_like(X)

        # Use bilinear interpolation of the PSF at the requested coordinates
        result[select] = interp2d(parameters["pixels"].value, pX[select], pY[select])

        return image.pixel_area * 10 ** result
