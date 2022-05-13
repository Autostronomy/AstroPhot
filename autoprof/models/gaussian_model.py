from .parametric_model_object import Parametric_Model


class Gaussian(Parametric_Model):

    model_type = "gaussian " + Parametric_Model.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
