from .parametric_model_object import Parametric_Model
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
from scipy.stats import iqr
from scipy.optimize import minimize

class Sersic(Parametric_Model):

    model_type = " ".join(("sersic", Parametric_Model.model_type))
    parameter_specs = {
        "Ie": {"units": "flux"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.05},
        "Re": {"units": "pix", "limits": (0,None)},
    }

    def initialize(self, target = None):
        super().initialize(target)
        if target is None:
            target = self.target
        if any((self["n"].value is None, self["Ie"].value is None, self["Re"].value is None)):
            # Get the sub-image area corresponding to the model image
            target_area = target.get_image_area(self.model_image)
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(self["center_x"].value, self["center_y"].value, target_area)
            
            iso_info = isophotes(
                target_area.data - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value, q = self["q"].value,
                n_isophotes = 6
            )
            R = np.array(list(iso["R"] for iso in iso_info))
            flux = np.array(list(iso["flux"] for iso in iso_info))
            x0 = [
                4. if self["n"].value is None else self["n"].value,
                R[1] if self["Re"].value is None else self["Re"].value,
                flux[1] if self["Ie"].value is None else self["Ie"].value,
            ]
            res = minimize(lambda x: np.mean((flux - sersic(R, x[0], x[1], x[2]))**2), x0 = x0,method = 'Nelder-Mead')
            for i, param in enumerate(["n", "Re", "Ie"]):
                if self[param].value is None:
                    self[param].set_value(res.x[i], override_fixed = True)
        if self["Re"].uncertainty is None:
            self["Re"].set_uncertainty(0.02 * self["Re"].value)
        if self["Ie"].uncertainty is None:
            self["Ie"].set_uncertainty(0.02 * np.abs(self["Ie"].value))

    def sample_model(self):
        super().sample_model()

        XX, YY = np.meshgrid(np.arange(self.window_shape[1],dtype = float), np.arange(self.window_shape[0], dtype=float))

        # Convert center coordinates to model image array indices
        icenter = coord_to_index(self["center_x"].value, self["center_y"].value, self.model_image)
        XX -= icenter[1]
        YY -= icenter[0]

        XX, YY = Rotate_Cartesian(-self["PA"].value, XX, YY)
        RR = np.sqrt(XX**2 + (YY / self["q"].value)**2)
        self.model_image += sersic(RR, self["n"].value, self["Re"].value, self["Ie"].value)
        
