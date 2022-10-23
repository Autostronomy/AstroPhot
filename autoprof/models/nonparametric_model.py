from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from autoprof.utils.interpolate import cubic_spline_torch
import numpy as np
from scipy.stats import binned_statistic, iqr
import torch

class NonParametric_Galaxy(Galaxy_Model):
    """basic galaxy model with a nonparametric radial light profile.

    """
    model_type = f"nonparametric {Galaxy_Model.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
    def set_fit_window(self, window):
        super().set_fit_window(window)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.fit_window.shape/2)**2)):
                self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
            self.profR.pop()
            self.profR = torch.tensor(self.profR)
            
    def initialize(self):
        super().initialize()
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
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        I = cubic_spline_torch(self.profR, self["I(R)"].value, R.view(-1)).view(*R.shape)
        res = 10**(I) * sample_image.pixelscale**2
        res[R > self.profR[-1]] = 0
        return res


class NonParametric_Warp(Warp_Galaxy):
    """warped coordinate galaxy model with a nonparametric light profile.

    """
    model_type = f"nonparametric {Warp_Galaxy.model_type}"
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
    
    def set_fit_window(self, window):
        super().set_fit_window(window)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.fit_window.shape/2)**2)):
                self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
            self.profR.pop()
            self.profR = torch.tensor(self.profR)
            
    def initialize(self):
        super().initialize()
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
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image
        I = cubic_spline_torch(self.profR, self["I(R)"].value, R.view(-1)).view(*R.shape)
        res = 10**(I) * sample_image.pixelscale**2
        res[R > self.profR[-1]] = 0
        return res
