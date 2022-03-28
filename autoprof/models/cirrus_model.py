from .sky_model_object import Sky_Model


class Cirrus(Sky_Model):

    name = "cirrus"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
