from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy

class Sersic_Galaxy(Galaxy_Model):
    """basic galaxy model with a sersic profile for the radial light
    profile.

    """
    model_type = f"sersic {Galaxy_Model.model_type}"
    parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
    
class Sersic_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a sersic profile for the
    radial light model.

    """
    model_type = f"sersic {Warp_Galaxy.model_type}"
    parameter_specs = {
        "I0": {"units": "flux/arcsec^2", "limits": (0,None)},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Rs": {"units": "arcsec", "limits": (0,None)},
    }

    from ._shared_methods import sersic_radial_model as radial_model
    from ._shared_methods import sersic_initialize as initialize
