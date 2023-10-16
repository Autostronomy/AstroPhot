from ..param import Param_Unlock, Param_SoftLimits
from .model_object import Component_Model
from .psf_model_object import PSF_Model
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target

__all__ = ("Point_Source",)

class Point_Source(Component_Model):
    """
    Describes a point source in the image, this is a delta function at some position in the sky. This is typically used to describe stars, supernovae, very small galaxies, quasars, asteroids or any other object which can essentially be entirely described by a position and total flux (no structure). 
    """
    model_type = f"pointsource {Component_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Component_Model._parameter_order + ("flux")
    useable = True

    # Psf convolution should be on by default since this is a delta function
    psf_mode = "full"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.psf is None:
            raise ValueError("Point_Source needs psf information")

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if parameters["flux"].value is not None:
            return
        target_area = target[self.window]
        with Param_Unlock(parameters["flux"]), Param_SoftLimits(parameters["flux"]):
            parameters["flux"].value = torch.log10(torch.sum(target_area.data))
            parameters["flux"].uncertainty = torch.std(target_area.data) / (torch.log(10) * 10**parameters["flux"].value)

    @default_internal
    def evaluate_model(
        self, X=None, Y=None, image=None, parameters: "Parameter_Node" = None, **kwargs
    ):
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]
            
        if isinstance(self.psf, PSF_Model):
            return self.psf.evaluate_model(X = X, Y = Y, image = image, parameters = parameters)
        elif isinstance(self.psf, PSF_Image):
            raise ValueError("Point_Source should have special evaluation for PSF_Image")
        
