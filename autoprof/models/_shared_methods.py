from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic, sersic_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import torch
from scipy.optimize import minimize

def sersic_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if any((self["n"].value is None, self["I0"].value is None, self["Rs"].value is None)):
            # Get the sub-image area corresponding to the model image
            target_area = self.target[self.fit_window]
            edge = np.concatenate((target_area.data[:,0], target_area.data[:,-1], target_area.data[0,:], target_area.data[-1,:]))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(
                self["center"].value[0].detach().item(),
                self["center"].value[1].detach().item(), target_area
            )
            iso_info = isophotes(
                target_area.data.detach().numpy(),
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().item(), q = self["q"].value.detach().item(),
                n_isophotes = 15
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
            flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
            if np.sum(flux < 0) > 1:
                flux -= np.min(flux)
            x0 = [
                2. if self["n"].value is None else self["n"].value.detach().item(),
                R[5] if self["Rs"].value is None else self["Rs"].value.detach().item(),
                flux[0] if self["I0"].value is None else self["I0"].value.detach().item(),
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = 'Nelder-Mead')
            for i, param in enumerate(["n", "Rs", "I0"]):
                self[param].set_value(res.x[i], override_locked = (self[param].value is None))
        if self["Rs"].uncertainty is None:
            self["Rs"].set_uncertainty(0.02 * self["Rs"].value.detach().item(), override_locked = True)
        if self["I0"].uncertainty is None:
            self["I0"].set_uncertainty(0.02 * np.abs(self["I0"].value.detach().item()), override_locked = True)


def sersic_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return sersic(R, self["n"].value, self["Rs"].value, self["I0"].value * sample_image.pixelscale**2)
