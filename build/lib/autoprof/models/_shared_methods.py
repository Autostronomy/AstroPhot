from ..utils.initialize import isophotes
from ..utils.parametric_profiles import sersic_torch, sersic_np, gaussian_torch, gaussian_np, exponential_torch, exponential_np, nonparametric_torch
from ..utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
from ..utils.conversions.functions import sersic_I0_to_flux_np, sersic_flux_to_I0_torch
from scipy.special import gamma
from scipy.stats import binned_statistic, iqr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr
import torch
from scipy.optimize import minimize

# Exponential
######################################################################
@torch.no_grad()
def exponential_initialize(self, target = None):
    if target is None:
        target = self.target
    super(self.__class__, self).initialize(target)
    if all((self["Ie"].value is not None, self["Re"].value is not None)):
        return
    # Get the sub-image area corresponding to the model image
    target_area = target[self.window]
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
    if self["Re"].value is None:
        self["Re"].set_value(res.x[1], override_locked = True)
    if self["Ie"].value is None:
        self["Ie"].set_value(np.log10(res.x[2]), override_locked = True)
    if self["Re"].uncertainty is None:
        self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().item(), override_locked = True)
    if self["Ie"].uncertainty is None:
        self["Ie"].set_uncertainty(0.02, override_locked = True)
            
def exponential_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.target
    return exponential_torch(R, self["Re"].value, (10**self["Ie"].value) * sample_image.pixelscale**2)

# Sersic
######################################################################
@torch.no_grad()
def sersic_initialize(self, target = None):
    if target is None:
        target = self.target
    super(self.__class__, self).initialize(target)
    if all((self["n"].value is not None, self["Ie"].value is not None, self["Re"].value is not None)):
        return
    # Get the sub-image area corresponding to the model image
    target_area = target[self.window]
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
        self["center"].value[0],
        self["center"].value[1], target_area
    )
    iso_info = isophotes(
        target_area.data.detach().cpu().numpy() - edge_average,
        (icenter[1].item(), icenter[0].item()),
        threshold = 3*edge_scatter,
        pa = self["PA"].value.detach().cpu().item() if "PA" in self else 0.,
        q = self["q"].value.detach().cpu().item() if "q" in self else 1.,
        n_isophotes = 15
    )
    R = (np.array(list(iso["R"] for iso in iso_info)) * self.target.pixelscale.item())
    flux = (np.array(list(iso["flux"] for iso in iso_info)) / self.target.pixelscale.item()**2)
    if np.sum(flux < 0) > 0:
        print("fixing flux")
        flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
    x0 = [
        2. if self["n"].value is None else self["n"].value.detach().cpu().item(),
        R[4] if self["Re"].value is None else self["Re"].value.detach().cpu().item(),
        flux[4],
    ]
    def optim(x):
        residual = (np.log10(flux) - np.log10(sersic_np(R, x[0], x[1], x[2])))**2
        N = np.argsort(residual)
        return np.mean(residual[:-3])
    res = minimize(optim, x0 = x0, method = "Nelder-Mead") # , bounds = ((0.5,6), (R[1]*1e-3, None), (flux[0]*1e-3, None))
    
    if self["n"].value is None:
        self["n"].set_value(res.x[0] if res.success else x0[0], override_locked = True)
    if self["Re"].value is None:
        self["Re"].set_value(res.x[1] if res.success else x0[1], override_locked = True)
    if self["Ie"].value is None:
        self["Ie"].set_value(np.log10(res.x[2] if res.success else x0[2]), override_locked = True)
    if self["Re"].uncertainty is None:
        self["Re"].set_uncertainty(0.02 * self["Re"].value.detach().cpu().item(), override_locked = True)
    if self["Ie"].uncertainty is None:
        self["Ie"].set_uncertainty(0.02, override_locked = True)
            
def sersic_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.target
    return sersic_torch(R, self["n"].value, self["Re"].value, (10**self["Ie"].value) * sample_image.pixelscale**2)

# Gaussian
######################################################################
@torch.no_grad()
def gaussian_initialize(self, target = None):
    if target is None:
        target = self.target
    super(self.__class__, self).initialize(target)
    if all((self["sigma"].value is not None, self["flux"].value is not None)):
        return
    # Get the sub-image area corresponding to the model image
    target_area = target[self.window]
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
    flux = np.array(list(iso["flux"] for iso in iso_info))
    if np.sum(flux < 0) >= 1:
        flux -= np.min(flux) - np.abs(np.min(flux)*0.1)
    x0 = [
        R[-1]/5 if self["sigma"].value is None else self["sigma"].value.detach().cpu().item(),
        np.log10(np.sum(target_area.data.detach().cpu().numpy()) if self["flux"].value is None else self["flux"].value.detach().cpu().item()),
    ]
    res = minimize(lambda x: np.mean((np.log10(flux) - np.log10(gaussian_np(R, x[0], 10**x[1])))**2), x0 = x0, method = "SLSQP", bounds = ((R[1]*1e-3, None), (flux[0]*1e-3, None))) #, method = 'Nelder-Mead'
    for i, param in enumerate(["sigma", "flux"]):
        if self[param].value is None:
            self[param].set_value(res.x[i], override_locked = True)
    if self["sigma"].uncertainty is None:
        self["sigma"].set_uncertainty(0.02 * self["sigma"].value.detach().cpu().item(), override_locked = True)
    if self["flux"].uncertainty is None:
        self["flux"].set_uncertainty(0.02 * np.abs(self["flux"].value.detach().cpu().item()), override_locked = True)

def gaussian_radial_model(self, R, sample_image = None):
    if sample_image is None:
        sample_image = self.target
    return gaussian_torch(R, self["sigma"].value, (10**self["flux"].value)*sample_image.pixelscale**2)

# NonParametric
######################################################################
@torch.no_grad()
def nonparametric_initialize(self, target = None):
    if target is None:
        target = self.target
    super(self.__class__, self).initialize(target)

    if self["I(R)"].value is not None:
        # Create the I(R) profile radii to match the input profile intensity values
        if self["I(R)"].prof is None:
            # create logarithmically spaced profile radii
            new_prof = [0] + list(np.logspace(
                np.log10(2*target.pixelscale),
                np.log10(np.sqrt(torch.sum((self.window.shape/2)**2).item())),
                len(self["I(R)"].value),
            ))
            new_prof.pop(-2)
            # ensure no step is smaller than a pixelscale
            for i in range(1,len(new_prof)):
                if new_prof[i] - new_prof[i-1] < target.pixelscale.item():
                    new_prof[i] = new_prof[i-1] + target.pixelscale.item()
            self["I(R)"].set_profile(new_prof)
        return
    
    # Create the I(R) profile radii if needed
    if self["I(R)"].prof is None:
        new_prof = [0,2*target.pixelscale]
        while new_prof[-1] < torch.max(self.window.shape/2):
            new_prof.append(new_prof[-1] + torch.max(2*target.pixelscale,new_prof[-1]*0.2))
        new_prof.pop()
        new_prof.pop()
        new_prof.append(torch.sqrt(torch.sum((self.window.shape/2)**2)))
        self["I(R)"].set_profile(new_prof)
        
    profR = self["I(R)"].prof.detach().cpu().numpy()
    target_area = target[self.window]
    X, Y = target_area.get_coordinate_meshgrid_torch(self["center"].value[0], self["center"].value[1])
    X, Y = self.transform_coordinates(X, Y)
    R = self.radius_metric(X, Y).detach().cpu().numpy()
    rad_bins = [profR[0]] + list((profR[:-1] + profR[1:])/2) + [profR[-1]*100]
    raveldat = target_area.data.detach().cpu().numpy().ravel()
    I = binned_statistic(R.ravel(), raveldat, statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale.item()**2
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
        sample_image = self.target
    return nonparametric_torch(R, self["I(R)"].prof, self["I(R)"].value, sample_image.pixelscale**2, extend = self.extend_profile)

def nonparametric_iradial_model(self, i, R, sample_image = None):
    if sample_image is None:
        sample_image = self.target
    return nonparametric_torch(R, self["I(R)"].prof, self["I(R)"].value[i], sample_image.pixelscale**2, extend = self.extend_profile)
