from .sky_model_object import Sky_Model
from autoprof.utils.calculations.agregate_pixel import _average, _scatter


class FlatSky(Sky_Model):

    name = "flat sky"
    parameter_names = Sky_Model.parameter_names + ("sky", "noise")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.fixed != "locked":
            if not "center" in self.fixed:
                self.fixed.update("center")

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
