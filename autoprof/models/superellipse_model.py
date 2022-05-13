from .ellipse_model import Ellipse


class SuperEllipse(Ellipse):

    model_type = "superellipse " + Ellipse.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
