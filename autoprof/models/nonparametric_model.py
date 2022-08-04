from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .isophote_model import Isophote_Galaxy
from .parameter_object import Parameter_Array
from autoprof.utils.initialize import isophotes
from autoprof.utils.parametric_profiles import sersic
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import binned_statistic, iqr

class NonParametric_Galaxy(Galaxy_Model):

    model_type = " ".join(("nonparametric", Galaxy_Model.model_type))
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_qualities = {
        "I(R)": {"form": "array", "loss": "global"},
    }

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if self["I(R)"].value is not None:
            return
            
        target_area = target[self.window]
        X, Y = target_area.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]            
        I = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], I[N])
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/2, bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], S[N])
        self["I(R)"].set_value(np.log10(np.abs(I)), override_fixed = True)
        self["I(R)"].set_uncertainty(S/(np.abs(I)*np.log(10)), override_fixed = True)
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value(), ext = "const", s = 0)
        return 10**(I(R)) * sample_image.pixelscale**2
            
class NonParametric_Warp(Warp_Galaxy):

    model_type = " ".join(("nonparametric", Warp_Galaxy.model_type))
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_qualities = {
        "I(R)": {"form": "array", "loss": "global"},
    }

    loss_speed_factor = 2

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if self["I(R)"].value is not None:
            return
            
        target_area = target[self.window]
        X, Y = target_area.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]            
        I = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], I[N])
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/2, bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], S[N])
        self["I(R)"].set_value(np.log10(np.abs(I)), override_fixed = True)
        self["I(R)"].set_uncertainty(S/(np.abs(I)*np.log(10)), override_fixed = True)
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value(), ext = "const", s = 0)
        return 10**(I(R)) * sample_image.pixelscale**2

    # def compute_loss(self, data):
    #     # If the image is locked, no need to compute the loss
    #     if self.locked:
    #         return

    #     super().compute_loss(data)

    #     X, Y = data.loss_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
    #     if self.loss_speed_factor != 1:
    #         X = X[::self.loss_speed_factor,::self.loss_speed_factor]
    #         Y = Y[::self.loss_speed_factor,::self.loss_speed_factor]
    #         dat = data.loss_image.data[::self.loss_speed_factor,::self.loss_speed_factor]
    #     else:
    #         dat = data.loss_image.data
    #     X, Y = self.transform_coordinates(X, Y)
    #     R = self.radius_metric(X, Y)
    #     reg = self.regularize_loss()
    #     rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            
    #     temp_loss = binned_statistic(R.ravel(), dat.ravel(), statistic = 'mean', bins = rad_bins)[0]
    #     N = np.isfinite(temp_loss)
    #     if not np.all(N):
    #         temp_loss[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], temp_loss[N])
                            
    #     self.loss["radial loss"] = np.array(temp_loss)


class NonParametric_Isophote(Isophote_Galaxy):

    model_type = " ".join(("nonparametric", Isophote_Galaxy.model_type))
    parameter_specs = {
        "I(R)": {"units": "log10(flux/arcsec^2)"},
    }
    parameter_qualities = {
        "I(R)": {"form": "array", "loss": "global"},
    }

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if self["I(R)"].value is not None:
            return
            
        target_area = target[self.window]
        X, Y = target_area.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]            
        I = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        N = np.isfinite(I)
        if not np.all(N):
            I[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], I[N])
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/2, bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], S[N])
        self["I(R)"].set_value(np.log10(np.abs(I)), override_fixed = True)
        self["I(R)"].set_uncertainty(S/(np.abs(I)*np.log(10)), override_fixed = True)
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value(), ext = "const", s = 0)
        return 10**(I(R)) * sample_image.pixelscale**2
        
