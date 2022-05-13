from .model_object import Model


class Star_Model(Model):

    model_type = "star " + Model.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
