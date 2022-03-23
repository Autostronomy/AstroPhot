from .sky_model_object import Sky_Model

class FlatSky(Sky_Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
