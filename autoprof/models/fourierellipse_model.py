from .ellipse_model import Ellipse


class FourierEllipse(Ellipse):

    model_type = "fourier " + Ellipse.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
