from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
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
        "I(R)": {"units": "flux/arcsec^2"},
    }
    parameter_qualities = {
        "I(R)": {"form": "array", "loss": "radial loss", "regularize": "self", "regularize scale": 1},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
        
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.window.shape/2)**2)):
                self.profR.append(max(1,self.profR[-1]*1.2))
            self.profR.pop()                
            self.profR = np.array(self.profR)
        
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
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/(2*np.sqrt(len(d))), bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], S[N])
        self["I(R)"].set_value(I, override_fixed = True)
        self["I(R)"].set_uncertainty(S, override_fixed = True)

    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(data)

        X, Y = data.loss_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        if self.loss_speed_factor != 1:
            X = X[::self.loss_speed_factor,::self.loss_speed_factor]
            Y = Y[::self.loss_speed_factor,::self.loss_speed_factor]        
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        reg = self._regularize_loss()
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            
        temp_loss = binned_statistic(R.ravel(), data.loss_image.data.ravel(), statistic = 'mean', bins = rad_bins)[0]
                            
        self.loss["radial loss"] = np.array(temp_loss)

        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value() * sample_image.pixelscale**2, ext = "zeros", s = 0)
        return I(R)

class SampleNonParametric_Warp(Warp_Galaxy):
    
    model_type = " ".join(("samplenonparametric", Warp_Galaxy.model_type))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profI = np.zeros(len(self.profR))
        self.sample_first = False
        
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        target_area = target[self.window]
        X, Y = target_area.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]            
        self.profI = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = 'median', bins = rad_bins)[0] / target_area.pixelscale**2
        
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image

        if self.sample_first:
            rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            self.profI = binned_statistic(R.ravel(), self.target[self.window].data.ravel(), statistic = 'median', bins = rad_bins)[0] / sample_image.pixelscale**2
            N = np.isfinite(self.profI)
            if not np.all(N):
                self.profI[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], self.profI[N])
            
        I = UnivariateSpline(self.profR, self.profI * sample_image.pixelscale**2, ext = "zeros", s = 0)
        return I(R)

    def sample_model(self, sample_image = None):

        if sample_image is None:
            sample_image = self.model_image

        if sample_image is self.model_image:
            self.sample_first = True
        super().sample_model(sample_image)
        self.sample_first = False
        
    
    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(data)

        X, Y = data.loss_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        if self.loss_speed_factor != 1:
            X = X[::self.loss_speed_factor,::self.loss_speed_factor]
            Y = Y[::self.loss_speed_factor,::self.loss_speed_factor]
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        reg = self._regularize_loss()
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            
        temp_loss = binned_statistic(R.ravel(), data.loss_image.data.ravel(), statistic = 'mean', bins = rad_bins)[0]
                            
        self.loss["radial loss"] = np.array(temp_loss)

        
class NonParametric_Warp(Warp_Galaxy):

    model_type = " ".join(("nonparametric", Warp_Galaxy.model_type))
    parameter_specs = {
        "I(R)": {"units": "flux/arcsec^2"},
    }
    parameter_qualities = {
        "I(R)": {"form": "array", "loss": "radial loss", "regularize": "self", "regularize scale": 1},
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
        S = binned_statistic(R.ravel(), target_area.data.ravel(), statistic = lambda d:iqr(d,rng=[16,84])/(2*np.sqrt(len(d))), bins = rad_bins)[0]
        N = np.isfinite(S)
        if not np.all(N):
            S[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], S[N])
        self["I(R)"].set_value(I, override_fixed = True)
        self["I(R)"].set_uncertainty(S, override_fixed = True)
        
    def radial_model(self, R, sample_image = None):
        if sample_image is None:
            sample_image = self.model_image        
        I = UnivariateSpline(self.profR, self["I(R)"].get_value() * sample_image.pixelscale**2, ext = "const", s = 0)
        return I(R)

    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(data)

        X, Y = data.loss_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        if self.loss_speed_factor != 1:
            X = X[::self.loss_speed_factor,::self.loss_speed_factor]
            Y = Y[::self.loss_speed_factor,::self.loss_speed_factor]
            dat = data.loss_image.data[::self.loss_speed_factor,::self.loss_speed_factor]
        else:
            dat = data.loss_image.data
        X, Y = self.transform_coordinates(X, Y)
        R = self.radius_metric(X, Y)
        reg = self._regularize_loss()
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            
        temp_loss = binned_statistic(R.ravel(), dat.ravel(), statistic = 'mean', bins = rad_bins)[0]
        N = np.isfinite(temp_loss)
        if not np.all(N):
            temp_loss[np.logical_not(N)] = np.interp(self.profR[np.logical_not(N)], self.profR[N], temp_loss[N])
                            
        self.loss["radial loss"] = np.array(temp_loss)
