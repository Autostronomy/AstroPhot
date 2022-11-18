from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
from autoprof.utils.interpolate import cubic_spline_torch, interp1d_torch
from autoprof.utils.conversions.functions import sersic_I0_to_flux_np, sersic_flux_to_I0_torch
from scipy.special import gamma
from scipy.stats import binned_statistic, iqr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import torch
from scipy.optimize import minimize

# Exponential
######################################################################
def exponential_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if all((self["Ie"].value is not None, self["Re"].value is not None)):
            return
        # Get the sub-image area corresponding to the model image
        target_area = self.target[self.fit_window]
        edge = np.concatenate((
            target_area.data.detach().cpu().numpy()[:,0],
            target_area.data.detach().cpu().numpy()[:,-1],
            target_area.data.detach().cpu().numpy()[0,:],
            target_area.data.detach().cpu().numpy()[-1,:]
        ))
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng = (16,84))/2
        # Convert center coordinates to target area array indices
        icenter = coord_to_index(
            self["center"].value[0].detach().cpu().item(),
            self["center"].value[1].detach().cpu().item(), target_area
        )
        iso_info = isophotes(
            target_area.data.detach().cpu().numpy() - edge_average,
            (icenter[1], icenter[0]),
            threshold = 3*edge_scatter,
            pa = self["PA"].value.detach().cpu().item() if "PA" in self else 0.,
            q = self["q"].value.detach().cpu().item() if "q" in self else 1.,
            n_isophotes = 15
        )
        R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
        flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
        if np.sum(flux < 0) >= 1:
            flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
        x0 = [
            R[4] if self["Re"].value is None else self["Re"].value.detach().cpu().item(),
            flux[4],
        ]
        res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(exponential_np(R, x[0], x[1])))**2), x0 = x0, method = "SLSQP", bounds = ((R[1]*1e-3, None), (flux[0]*1e-3, None)))
        self["Re"].set_value(res.x[1], override_locked = self["Re"].value is None)
        self["Ie"].set_value(np.log10(res.x[2]), override_locked = (self["Ie"].value is None))
        if self["Re"].uncertainty is None:
            self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().item(), override_locked = True)
        if self["Ie"].uncertainty is None:
            self["Ie"].set_uncertainty(0.02, override_locked = True)
            
def exponential_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return exponential_torch(R, self["Re"].value, (10**self["Ie"].value) * sample_image.pixelscale**2)

# Sersic
######################################################################
def sersic_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if all((self["n"].value is not None, self["Ie"].value is not None, self["Re"].value is not None)):
            return
        # Get the sub-image area corresponding to the model image
        target_area = self.target[self.fit_window]
        edge = np.concatenate((
            target_area.data.detach().cpu().numpy()[:,0],
            target_area.data.detach().cpu().numpy()[:,-1],
            target_area.data.detach().cpu().numpy()[0,:],
            target_area.data.detach().cpu().numpy()[-1,:]
        ))
        edge_average = np.median(edge)
        edge_scatter = iqr(edge, rng = (16,84))/2
        # Convert center coordinates to target area array indices
        icenter = coord_to_index(
            self["center"].value[0].detach().cpu().item(),
            self["center"].value[1].detach().cpu().item(), target_area
        )
        iso_info = isophotes(
            target_area.data.detach().cpu().numpy() - edge_average,
            (icenter[1], icenter[0]),
            threshold = 3*edge_scatter,
            pa = self["PA"].value.detach().cpu().item() if "PA" in self else 0.,
            q = self["q"].value.detach().cpu().item() if "q" in self else 1.,
            n_isophotes = 15
        )
        R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
        flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
        if np.sum(flux < 0) >= 1:
            flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
        x0 = [
            2. if self["n"].value is None else self["n"].value.detach().cpu().item(),
            R[4] if self["Re"].value is None else self["Re"].value.detach().cpu().item(),
            flux[4],
        ]
        res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2), x0 = x0, method = "SLSQP", bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None)))
        self["n"].set_value(res.x[0], override_locked = self["n"].value is None)
        self["Re"].set_value(res.x[1], override_locked = self["Re"].value is None)
        self["Ie"].set_value(np.log10(res.x[2]), override_locked = (self["Ie"].value is None))
        if self["Re"].uncertainty is None:
            self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().item(), override_locked = True)
        if self["Ie"].uncertainty is None:
            self["Ie"].set_uncertainty(0.02, override_locked = True)
            
def sersic_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return sersic_torch(R, self["n"].value, self["Re"].value, (10**self["Ie"].value) * sample_image.pixelscale**2)

# Gaussian
######################################################################
def gaussian_initialize(self):
    super(self.__class__, self).initialize()
    with torch.no_grad():
        if any((self["sigma"].value is None, self["flux"].value is None)):
            # Get the sub-image area corresponding to the model image
            target_area = self.target[self.fit_window]
            edge = np.concatenate((
                target_area.data.detach().cpu().numpy()[:,0],
                target_area.data.detach().cpu().numpy()[:,-1],
                target_area.data.detach().cpu().numpy()[0,:],
                target_area.data.detach().cpu().numpy()[-1,:]
            ))
            edge_average = np.median(edge)
            edge_scatter = iqr(edge, rng = (16,84))/2
            # Convert center coordinates to target area array indices
            icenter = coord_to_index(
                self["center"].value[0].detach().cpu().item(),
                self["center"].value[1].detach().cpu().item(), target_area
            )
            iso_info = isophotes(
                target_area.data.detach().cpu().numpy() - edge_average,
                (icenter[1], icenter[0]),
                threshold = 3*edge_scatter,
                pa = self["PA"].value.detach().cpu().item(), q = self["q"].value.detach().cpu().item(),
                n_isophotes = 15
            )
            R = np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale
            flux = np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale**2
            if np.sum(flux < 0) >= 1:
                flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
            x0 = [
                R[-1]/5 if self["sigma"].value is None else self["sigma"].value.detach().cpu().item(),
                np.sum(target_area.data.detach().cpu().numpy()) if self["flux"].value is None else self["flux"].value.detach().cpu().item(),
            ]
            res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(gaussian_np(R, x[0], x[1])))**2), x0 = x0, method = "SLSQP", bounds = ((R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
            for i, param in enumerate(["sigma", "flux"]):
                self[param].set_value(res.x[i], override_locked = self[param].value is None)
        if self["sigma"].uncertainty is None:
            self["sigma"].set_uncertainty(0.02 * self["sigma"].value.detach().cpu().item(), override_locked = True)
        if self["flux"].uncertainty is None:
            self["flux"].set_uncertainty(0.02 * np.abs(self["flux"].value.detach().cpu().item()), override_locked = True)

def gaussian_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    return gaussian_torch(R, self["sigma"].value, self["flux"].value * sample_image.pixelscale**2)

# NonParametric
######################################################################

def nonparametric_set_fit_window(self, window):
    super(self.__class__, self).set_fit_window(window)
    
    if self.profR is None:
        self.profR = [0,self.target.pixelscale]
        while self.profR[-1] < np.min(self.fit_window.shape/2):
            self.profR.append(self.profR[-1] + max(self.target.pixelscale,self.profR[-1]*0.2))
        self.profR.pop()
        self.profR.append(np.sqrt(np.sum((self.fit_window.shape/2)**2)))
        self.profR = torch.tensor(self.profR, dtype = self.dtype, device = self.device)
            
def nonparametric_initialize(self):
    super(self.__class__, self).initialize()
    if self["I(R)"].value is not None:
        return
    
    with torch.no_grad():
        profR = self.profR.detach().cpu().numpy()
        target_area = self.target[self.fit_window]
        X, Y = target_area.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y).detach().cpu().numpy()
        rad_bins = [profR[0]] + list((profR[:-1] + profR[1:])/2) + [profR[-1]*100]
        raveldat = target_area.data.detach().cpu().numpy().ravel()
        I = binned_statistic(R.ravel(), raveldat, statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], I[N])
        S = binned_statistic(R.ravel(), raveldat, statistic = lambda d:iqr(d,rng=[16,84])/2, bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(profR[np.logical_not(N)], profR[N], S[N])
        self["I(R)"].set_value(np.log10(np.abs(I)), override_locked = True)
        self["I(R)"].set_uncertainty(S/(np.abs(I)*np.log(10)), override_locked = True)

def nonparametric_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.model_image
    I =  cubic_spline_torch(self.profR, self["I(R)"].value, R.view(-1), extend = "none").view(*R.shape) # interp1d_torch(self.profR, self["I(R)"].value, R)
    res = 10**(I) * sample_image.pixelscale**2
    res[R > self.profR[-1]] = 0
    return res
