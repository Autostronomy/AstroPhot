from .model_object import Model
import numpy as np

class Parametric_Model(Model):

    model_type = "parametric " + Model.model_type
    parameter_specs = {**Model.parameter_specs, **{
        "q": {"units": "b/a", "limits": (0,1)},
        "PA": {"units": "rad", "limits": (0,np.pi), "cyclic": True},
    }}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()
        init = isophote_initialize(
            self.image[
                self.window[1][0] : self.window[1][1],
                self.window[1][0] : self.window[1][1],
            ],
            n_isophotes=3,
        )

        if self.get_value("q") is None:
            self.set_value("q", np.mean(init["q"]))
        if self.get_value("PA") is None:
            self.set_value("PA", np.mean(init["PA"]))
        
