from .sky_model_object import Sky_Model
from autoprof.utils.calculations.agregate_pixel import _average, _scatter


class FlatSky(Sky_Model):

    model_type = "flat " + Sky_Model.model_type
    parameter_specs = {**Sky_Model.parameter_specs, **{
        "sky": {"units": "flux"},
        "noise": {"units": "flux", "limits": (0,None)},
    }}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()
        if self.get_value("sky") is None:
            self.set_value(
                "sky",
                _average(
                    self.image[
                        self.window[1][0] : self.window[1][1],
                        self.window[1][0] : self.window[1][1],
                    ]
                ),
                override_fixed=True,
            )
        if self.get_value("noise") is None:
            self.set_value(
                "noise",
                _scatter(
                    self.image[
                        self.window[1][0] : self.window[1][1],
                        self.window[1][0] : self.window[1][1],
                    ]
                ),
                override_fixed=True,
            )
