import torch
import numpy as np

from .psf_model_object import PSF_Model
from ..image import PSF_Image
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import interp2d
from ._shared_methods import select_target
from ..param import Param_Unlock, Param_SoftLimits
from .. import AP_config

__all__ = ["Eigen_PSF"]

class Eigen_PSF(PSF_Model):
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

    model_type = f"eigen {PSF_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux/arcsec^2)", "value": 0., "locked": True},
        "weights": {"units": "unitless"},
    }
    _parameter_order = PSF_Model._parameter_order + ("flux","weights")
    useable = True
    model_integrated = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "eigen_basis" not in kwargs:
            AP_config.ap_logger.warning("Eigen basis not supplied! Assuming psf as single basis element. Please provide Eigen basis or just use an empirical PSF image.")
            self.eigen_basis = torch.clone(self.target.data).unsqueeze(0)
            self.parameters["weights"].locked = True
        else:
            self.eigen_basis = torch.as_tensor(
                kwargs["eigen_basis"],
                dtype = AP_config.ap_dtype,
                device = AP_config.ap_device
            )
        if kwargs.get("normalize_eigen_basis", True):
            self.eigen_basis = self.eigen_basis / torch.sum(self.eigen_basis, axis = (1,2)).unsqueeze(1).unsqueeze(2)
        self.eigen_pixelscale = torch.as_tensor(
            kwargs.get("eigen_pixelscale", 1. if self.target is None else self.target.pixelscale),
            dtype = AP_config.ap_dtype,
            device = AP_config.ap_device
        )
        
    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)
        target_area = target[self.window]
        with Param_Unlock(parameters["flux"]), Param_SoftLimits(parameters["flux"]):
            if parameters["flux"].value is None:
                parameters["flux"].value = torch.log10(torch.abs(torch.sum(target_area.data)) / target.pixel_area)
            if parameters["flux"].uncertainty is None:
                parameters["flux"].uncertainty = torch.abs(parameters["flux"].value) * self.default_uncertainty
        with Param_Unlock(parameters["weights"]), Param_SoftLimits(parameters["weights"]):
            if parameters["weights"].value is None:
                W = np.zeros(len(self.eigen_basis))
                W[0] = 1.
                parameters["weights"].value = W
            if parameters["weights"].uncertainty is None:
                parameters["weights"].uncertainty = torch.ones_like(parameters["weights"].value) * self.default_uncertainty

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]

        psf_model = PSF_Image(
            data = torch.clamp(torch.sum(self.eigen_basis.detach() * (parameters["weights"].value / torch.linalg.norm(parameters["weights"].value)).unsqueeze(1).unsqueeze(2), axis = 0), min = 0.),
            pixelscale = self.eigen_pixelscale.detach(),
        )
        
        # Convert coordinates into pixel locations in the psf image
        pX, pY = psf_model.plane_to_pixel(X, Y)

        # Select only the pixels where the PSF image is defined
        select = torch.logical_and(
            torch.logical_and(pX > -0.5, pX < psf_model.data.shape[1]-0.5),
            torch.logical_and(pY > -0.5, pY < psf_model.data.shape[0]-0.5),
        )

        # Zero everywhere outside the psf
        result = torch.zeros_like(X)

        # Use bilinear interpolation of the PSF at the requested coordinates
        result[select] = interp2d(psf_model.data, pX[select], pY[select])

        # Ensure positive values
        result = torch.clamp(result, min = 0.)
        
        return result * (image.pixel_area * 10 ** parameters["flux"].value)
