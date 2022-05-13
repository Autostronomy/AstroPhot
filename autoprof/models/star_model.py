from .star_model_object import Star_Model
import numpy as np

class Star(Star_Model):

    model_type = "psf " + Star_Model.model_type
    parameter_specs = {**Star_Model.parameter_specs, **{
        "A": {"units": "flux"},
    }}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()
        if self.get_value("Flux") is None:
            self.set_value(
                "Flux",
                np.sum(
                    self.image[
                        self.window[1][0] : self.window[1][1],
                        self.window[0][0] : self.window[0][1],
                    ]
                ),
            )
