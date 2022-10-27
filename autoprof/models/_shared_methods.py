from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import torch
from scipy.optimize import minimize

# Sersic
######################################################################
def sersic_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if all((self["n"].value is not None, self["I0"].value is not None, self["Rs"].value is not None)):
            return
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
            target_area.data.detach().numpy() - edge_average,
            (icenter[1], icenter[0]),
            threshold = 3*edge_scatter,
            pa = self["PA"].value.detach().item(), q = self["q"].value.detach().item(),
            n_isophotes = 15
        )
        R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
        flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
        if np.sum(flux < 0) >= 1:
            flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
        x0 = [
            2. if self["n"].value is None else self["n"].value.detach().item(),
            R[1] if self["Rs"].value is None else self["Rs"].value.detach().item(),
            flux[0] if self["I0"].value is None else self["I0"].value.detach().item(),
        ]
        res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
        plt.scatter(R, np.log10(flux))
        plt.plot(R, np.log10(sersic_np(R, res.x[0], res.x[1], res.x[2])), color = 'r', label = 'fit')
        plt.plot(R, np.log10(sersic_np(R, x0[0], x0[1], x0[2])), color = 'orange', label = 'init')
        plt.legend()
        plt.title(f"{res.success} n {res.x[0]:0.3f} Rs {res.x[1]:0.3e} I0 {res.x[2]:0.3e}")
        plt.savefig(f"{self.name}_coma_test.jpg")
        plt.close()
        for i, param in enumerate(["n", "Rs", "I0"]):
            self[param].set_value(res.x[i], override_locked = (self[param].value is None))
        if self["Rs"].uncertainty is None:
            self["Rs"].set_uncertainty(0.02 * self["Rs"].value.detach().item(), override_locked = True)
        if self["I0"].uncertainty is None:
            self["I0"].set_uncertainty(0.02 * np.abs(self["I0"].value.detach().item()), override_locked = True)


def sersic_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return sersic_torch(R, self["n"].value, self["Rs"].value, self["I0"].value * sample_image.pixelscale**2)

# Gaussian
######################################################################
def gaussian_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if any((self["sigma"].value is None, self["flux"].value is None)):
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
                target_area.data.detach().numpy() - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().item(), q = self["q"].value.detach().item(),
                n_isophotes = 15
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
            flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
            if np.sum(flux < 0) >= 1:
                flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
            x0 = [
                R[-1]/5 if self["sigma"].value is None else self["sigma"].value.detach().item(),
                np.sum(target_area.data.detach().numpy()) if self["flux"].value is None else self["flux"].value.detach().item(),
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(gaussian_np(R, x[0], x[1])))**2), x0 = x0, method = "SLSQP", bounds = ((R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
            plt.scatter(R, np.log10(flux))
            plt.plot(R, np.log10(gaussian_np(R, res.x[0], res.x[1])), color = 'r', label = 'fit')
            plt.plot(R, np.log10(gaussian_np(R, x0[0], x0[1])), color = 'orange', label = 'init')
            plt.legend()
            plt.title(f"{res.success} sigma {res.x[0]:0.3f} I0 {res.x[1]:0.3e}")
            plt.savefig(f"{self.name}_coma_gaussian_test.jpg")
            plt.close()
            for i, param in enumerate(["sigma", "flux"]):
                self[param].set_value(res.x[i], override_locked = self[param].value is None)
        if self["sigma"].uncertainty is None:
            self["sigma"].set_uncertainty(0.02 * self["sigma"].value.detach().item(), override_locked = True)
        if self["flux"].uncertainty is None:
            self["flux"].set_uncertainty(0.02 * np.abs(self["flux"].value.detach().item()), override_locked = True)

def gaussian_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return gaussian_torch(R, self["sigma"].value, self["flux"].value * sample_image.pixelscale**2)

# NonParametric
######################################################################

def nonparametric_set_fit_window(self, window):
    super(self.__class__, self).set_fit_window(window)
    
    if self.profR is None:
        self.profR = [0,1]
        while self.profR[-1] < np.sqrt(np.sum((self.fit_window.shape/2)**2)): # fixme pixelscale vs image scale
            self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
        self.profR.pop()
        self.profR = torch.tensor(self.profR)
            
def nonparametric_initialize(self):
    super(self.__class__, self).initialize()
    if self["I(R)"].value is not None:
        return
    
    with torch.no_grad():
        profR = self.profR.detach().numpy()
        target_area = self.target[self.fit_window]
        X, Y = target_area.get_coordinate_meshgrid_np(self["center"].value[0].detach().item(), self["center"].value[1].detach().item())
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y).detach().numpy()
        rad_bins = [profR[0]] + list((profR[:-1] + profR[1:])/2) + [profR[-1]*100]
        raveldat = target_area.data.detach().numpy().ravel()
        I = binned_statistic(R.ravel(), raveldat, statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/2, bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
        self["I(R)"].set_value(np.log10(np.abs(I)), override_locked = True)
        self["I(R)"].set_uncertainty(S/(np.abs(I)*np.log(10)), override_locked = True)

def nonparametric_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    I = cubic_spline_torch(self.profR, self["I(R)"].value, R.view(-1)).view(*R.shape)
    res = 10**(I) * sample_image.pixelscale**2
    res[R > self.profR[-1]] = 0
    return res
