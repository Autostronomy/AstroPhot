from .model_object import Model


class NonParametric_Model(Model):

    model_type = "nonparametric " + Model.model_type
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
