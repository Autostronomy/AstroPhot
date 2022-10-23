from .model_object import BaseModel

class Sky_Model(BaseModel):
    """prototype class for any sky backgorund model.

    """
    model_type = f"sky {BaseModel.model_type}"
    psf_mode = 'none'
    parameter_specs = {
        "center": {"units": "arcsec", "locked": True, "uncertainty": 0.0},
    }
