from .parametric_model_object import Parametric_Model
from autoprof.utils.initialize_functions import isophote_initialize
from autoprof.utils.calculations.profiles import sersic
from autoprof.utils.image_operations import rotate_coordinates

class Sersic(Parametric_Model):

    model_type = " ".join(("sersic", Parametric_Model.model_type))
    parameter_specs = {
        "Ie": {"units": "flux"},
        "n": {"units": "none", "limits": (0,8), "uncertainty": 0.03},
        "Re": {"units": "pix", "limits": (0,None)},
    }

    def initialize(self):
        super().initialize()

        if any(self["n"].value is None, self["Ie"].value is None, self["Re"].value is None):
            iso_info = isophote_initialize(
                self.image - np.median(self.image),
                self["center"].value,
                threshold = 3*iqr(self.image, (16,84))/2,
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
            res = minimize(lambda x: np.mean((flux - sersic(R, x[0], x[1], x[2]))**2), x0 = x0)
            for i, param in enumerate(["n", "Re", "Ie"]):
                if self[param].value is None:
                    self[param].set_value(res.x[i], override_fixed = True)
        if self["Re"].uncertainty is None:
            self["Re"].set_uncertainty(0.02 * self["Re"])
        if self["Ie"].uncertainty is None:
            self["Ie"].set_uncertainty(0.02 * self["Ie"])

    def sample_model(self):
        super().sample_model()

        XX, YY = np.meshgrid(range(self.window[1][1] - self.window[1][0]), range(self.window[0][1] - self.window[0][0]))

        XX -= self["center"].value[0]
        YY -= self["center"].value[1]

        XX, YY = rotate_coordinates(XX, YY, self["PA"].value)
        RR = np.sqrt(XX**2 + (YY / self["q"].value)**2)

        self.model_image += sersic(RR, self["n"].value, self["Re"].value, self["Ie"].value)

        
