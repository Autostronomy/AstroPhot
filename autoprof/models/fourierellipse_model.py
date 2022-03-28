from .nonparametric_model_object import NonParametric_Model


class FourierEllipse(NonParametric_Model):

    name = "nonparametric fourier ellipse"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
