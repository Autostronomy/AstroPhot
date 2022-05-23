from .sky_model_object import Sky_Model
from autoprof.utils.calculations.agregate_pixel import _average, _scatter
import numpy as np
from scipy.stats import iqr

class FlatSky(Sky_Model):

    model_type = " ".join(("flat", Sky_Model.model_type))
    parameter_specs = {
        "sky": {"units": "flux"},
        "noise": {"units": "flux", "limits": (0,None)},
    }
    
    def initialize(self):        
        super().initialize()
        
        if self["sky"].value is None:
            self["sky"].set_value(
                float(np.median(self.image)),
                override_fixed=True,
            )
        if self["sky"].uncertainty is None:
            self["sky"].set_uncertainty(
                (iqr(self.image, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0) / np.sqrt(self.image.shape[0] * self.image.shape[1]),
                override_fixed=True,
            )
        if self["noise"].value is None:
            self["noise"].set_value(
                iqr(self.image, rng=(31.731 / 2, 100 - 31.731 / 2)) / 2.0,
                override_fixed=True,
            )
        if self["noise"].uncertainty is None:
            self["noise"].set_uncertainty(
                self["noise"].value / np.sqrt(2 * self.image.shape[0] * self.image.shape[1] - 2),
                override_fixed=True,
            )

    def sample_model(self):
        super().sample_model()

        self.model_image += self["sky"].value
