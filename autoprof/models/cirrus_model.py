from .sky_model_object import Sky_Model


class Cirrus(Sky_Model):

    model_type = "cirrus " + Sky_Model.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
