from .parametric_model_object import Parametric_Model


class Gaussian(Parametric_Model):

    name = "gaussian"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
