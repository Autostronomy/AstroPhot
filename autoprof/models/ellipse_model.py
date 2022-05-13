from .nonparametric_model_object import NonParametric_Model


class Ellipse(NonParametric_Model):

    model_type = "ellipse " + NonParametric_Model.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
