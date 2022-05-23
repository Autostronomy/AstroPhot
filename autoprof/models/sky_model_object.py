from .model_object import Model


class Sky_Model(Model):

    model_type = " ".join(("sky", Model.model_type))
    parameter_specs = {
        "center_x": {"units": "pix", "fixed": True, "uncertainty": 0.0},
        "center_y": {"units": "pix", "fixed": True, "uncertainty": 0.0},
    }
