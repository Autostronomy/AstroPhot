from ..param import Param_Unlock, Param_SoftLimits
from .model_object import Component_Model
from .psf_model_object import PSF_Model
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target

__all__ = ("Point_Source",)

class Point_Source(Component_Model):

    model_type = f"pointsource {Component_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Component_Model._parameter_order + ("flux")
    useable = True

    psf_mode = "full"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if isinstance(kwargs["psf"], PSF_Image) or isinstance(kwargs["psf"], PSF_Model):
            self.psf = kwargs["psf"]
        else:
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
        
