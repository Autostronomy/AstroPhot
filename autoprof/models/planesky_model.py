from .sky_model_object import Sky_Model


class PlaneSky(Sky_Model):

    model_type = "plane " + Sky_Model.model_type

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
