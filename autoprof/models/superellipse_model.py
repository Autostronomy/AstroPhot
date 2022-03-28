from .nonparametric_model_object import NonParametric_Model


class SuperEllipse(NonParametric_Model):

    name = "nonparametric superellipse"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
