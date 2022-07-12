from .model_object import BaseModel

class Sky_Model(BaseModel):

    model_type = " ".join(("sky", BaseModel.model_type))
    psf_mode = 'none'
    sample_mode = "direct"
    parameter_specs = {
        "center": {"units": "arcsec", "fixed": True, "uncertainty": 0.0},
    }
