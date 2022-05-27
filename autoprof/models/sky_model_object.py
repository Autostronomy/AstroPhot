from .model_object import BaseModel

class Sky_Model(BaseModel):

    model_type = " ".join(("sky", BaseModel.model_type))
    PSF_mode = 'none'
    parameter_specs = {
        "center_x": {"units": "pix", "fixed": True, "uncertainty": 0.0},
        "center_y": {"units": "pix", "fixed": True, "uncertainty": 0.0},
    }
