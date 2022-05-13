from .model_object import Model

class Constrained_Model(Model):

    model_type = "constrained " + Model.model_type

    def __init__(self, models, **kwargs):
        super().__init__(**kwargs)

        self.models = models
