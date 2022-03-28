from .star_model_object import Star_Model
import numpy as np


class Star(Star_Model):

    name = "star"
    parameter_names = Star_Model.parameter_names + ("F",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()
        if self.get_value("F") is None:
            self.set_value(
                "F",
                np.sum(
                    self.image[
                        self.window[1][0] : self.window[1][1],
                        self.window[0][0] : self.window[0][1],
                    ]
                ),
            )
