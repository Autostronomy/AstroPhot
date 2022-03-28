from .model_object import Model


class Star_Model(Model):

    name = "star model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
