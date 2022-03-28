from .parametric_model_object import Parametric_Model
from autoprof.utils.isophote_operations import isophote_initialize
from autoprof.utils.calculations.profiles import sersic


class Sersic(Parametric_Model):

    name = "sersic"
    parameter_names = Parametric_Model.parameter_names + ("Ie", "n", "Re", "q", "PA")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()
        if self.get_value("n") is None:
            self.set_value("n", 4.0)
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

        res = minimize(
            lambda i, r: np.mean(
                (init["f"] - sersic(init["R"], n=self.get_value("n"), Re=r, Ie=i)) ** 2
            ),
            x0=[init["f"][1], init["R"][1]],
            method="Nelder-Mead",
        )

        if self.get_value("Ie") is None:
            self.set_value("Ie", np.mean(res.x[0]))
        if self.get_value("Re") is None:
            self.set_value("Re", np.mean(res.x[1]))
