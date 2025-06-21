import torch
import numpy as np

from .psf_model_object import PSFModel
from ..image import PSFImage
from ..utils.decorators import ignore_numpy_warnings
from ..utils.interpolate import interp2d
from .. import AP_config
from ..errors import SpecificationConflict

__all__ = ["EigenPSF"]


class EigenPSF(PSFModel):
    """point source model which uses multiple images as a basis for the
    PSF as its representation for point sources. Using bilinear
    interpolation it will shift the PSF within a pixel to accurately
    represent the center location of a point source. There is no
    functional form for this object type as any image can be
    supplied. Note that as an argument to the model at construction
    one can provide "psf" as an AstroPhot PSF_Image object. Since only
    bilinear interpolation is performed, it is recommended to provide
    the PSF at a higher resolution than the image if it is near the
    nyquist sampling limit. Bilinear interpolation is very fast and
    accurate for smooth models, so this way it is possible to do the
    expensive interpolation before optimization and save time. Note
    that if you do this you must provide the PSF as a PSF_Image object
    with the correct pixelscale (essentially just divide the
    pixelscale by the upsampling factor you used).

    Args:
      eigen_basis (tensor): This is the basis set of images used to form the eigen point source, it should be a tensor with shape (N x W x H) where N is the number of eigen images, and W/H are the dimensions of the image.
      eigen_pixelscale (float): This is the pixelscale associated with the eigen basis images.

    Parameters:
        flux: the total flux of the point source model, represented as the log of the total flux.
        weights: the relative amplitude of the Eigen basis modes.

    """

    _model_type = "eigen"
    _parameter_specs = {
        "flux": {"units": "flux/arcsec^2", "value": 1.0},
        "weights": {"units": "unitless"},
    }
    usable = True

    def __init__(self, *args, eigen_basis=None, **kwargs):
        super().__init__(*args, **kwargs)
        if eigen_basis is None:
            raise SpecificationConflict(
                "EigenPSF model requires 'eigen_basis' argument to be provided."
            )
        self.eigen_basis = torch.as_tensor(
            kwargs["eigen_basis"],
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()
        target_area = self.target[self.window]
        if self.flux.value is None:
            self.flux.dynamic_value = (
                torch.abs(torch.sum(target_area.data)) / target_area.pixel_area
            )
            self.flux.uncertainty = self.flux.value * self.default_uncertainty
        if self.weights.value is None:
            self.weights.dynamic_value = 1 / np.arange(len(self.eigen_basis))
            self.weights.uncertainty = self.weights.value * self.default_uncertainty

    def brightness(self, x, y, flux, weights):
        x, y = self.transform_coordinates(x, y)

        psf = torch.sum(
            self.eigen_basis * (weights / torch.linalg.norm(weights)).unsqueeze(1).unsqueeze(2),
            axis=0,
        )

        pX, pY = self.target.plane_to_pixel(x, y)
        result = interp2d(psf, pX, pY)

        return result * flux
