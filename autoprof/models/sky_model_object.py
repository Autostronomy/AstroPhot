from .model_object import Model


class Sky_Model(Model):

    name = "sky model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
