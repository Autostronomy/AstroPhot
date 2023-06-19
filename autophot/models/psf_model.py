import torch

from .star_model_object import Star_Model
from ..image import Model_Image
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import interp2d
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["PSF_Star"]


class PSF_Star(Star_Model):
    """Star model which uses an image of the PSF as it's representation
    for stars. Using bilinear interpolation it will shift the PSF
    within a pixel to accurately represent the center location of a
    point source. There is no funcitonal form for this object type as
    any image can be supplied. Note that as an argument to the model
    at construction one can provide "psf" as an AutoPhot Model_Image
    object. Since only bilinear interpolation is performed, it is
    recommended to provide the PSF at a higher resolution than the
    image if it is near the nyquist sampling limit. Bilinear
    interpolation is very fast and accurate for smooth models, so this
    way it is possible to do the expensive interpolation before
    optimization and save time. Note that if you do this you must
    provide the PSF as a Model_Image object with the correct PSF
    (essentially just divide the PSF by the upsampling factor you
    used).

    Parameters:
        flux: the total flux of the star model, represented as the log of the total flux.

    """

    model_type = f"psf {Star_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("flux",)
    useable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fixme, model already has PSF interface, those can just be merged
        if "psf" in kwargs:
            self.psf_model = kwargs["psf"]
        else:
            self.psf_model = Model_Image(
                data=torch.clone(self.psf.data),
                pixelscale=self.psf.pixelscale,
            )
        self.psf_model.header.shift_origin(
            self.psf_model.origin - self.psf_model.center
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)
        target_area = target[self.window]
        if parameters["flux"].value is None:
            parameters["flux"].set_value(
                torch.log10(torch.abs(torch.sum(target_area.data)) / target.pixel_area),
                override_locked=True,
            )
        if parameters["flux"].uncertainty is None:
            parameters["flux"].set_uncertainty(
                torch.abs(parameters["flux"].value) * 1e-2, override_locked=True
            )

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]

        # Convert coordinates into pixel locations in the psf image
        pX, pY = self.psf_model.world_to_pixel(
            torch.stack((X, Y)).view(2, -1), unsqueeze_origin=True
        )
        pX = pX.reshape(X.shape)
        pY = pY.reshape(Y.shape)

        # Select only the pixels where the PSF image is defined
        select = torch.logical_and(
            torch.logical_and(pX > -0.5, pX < self.psf_model.data.shape[1]),
            torch.logical_and(pY > -0.5, pY < self.psf_model.data.shape[0]),
        )

        # Zero everywhere outside the psf
        result = torch.zeros_like(X)

        # Use bilinear interpolation of the PSF at the requested coordinates
        result[select] = interp2d(self.psf_model.data, pX[select], pY[select])

        return result * (image.pixel_area * 10 ** parameters["flux"].value)
