from .parametric_model_object import Parametric_Model

class Gaussian(Parametric_Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
