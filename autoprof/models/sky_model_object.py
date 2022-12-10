from .model_object import BaseModel

__all__ = ["Sky_Model"]

class Sky_Model(BaseModel):
    """prototype class for any sky backgorund model. This simply imposes
    that the center is a locked parameter, not involved in the fit.

    """
    model_type = f"sky {BaseModel.model_type}"
    psf_mode = "none"
    integrate_mode = "none"
    parameter_specs = {
        "center": {"units": "arcsec", "locked": True, "uncertainty": 0.0},
    }
