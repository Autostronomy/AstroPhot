from .model_object import Model


class Sky_Model(Model):

    model_type = "sky " + Model.model_type
    parameter_specs = {**Model.parameter_specs, **{
        "center": {"units": "pix", "fixed": True},
    }}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
