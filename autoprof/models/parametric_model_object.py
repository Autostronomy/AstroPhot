from .model_object import Model


class Parametric_Model(Model):

    name = "parametric model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
