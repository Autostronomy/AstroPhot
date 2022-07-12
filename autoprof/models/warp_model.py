from .galaxy_model_object import Galaxy_Model
from .parameter_object import Parameter_Array
import numpy as np
from scipy.interpolate import UnivariateSpline
from autoprof.utils.initialize import isophotes
from autoprof.utils.interpolate import nearest_neighbor
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, Axis_Ratio_Cartesian, coord_to_index, index_to_coord
from scipy.fftpack import fft, ifft
from scipy.stats import iqr, binned_statistic, binned_statistic_2d
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize

class Warp_Galaxy(Galaxy_Model):

    model_type = " ".join(("warp", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    parameter_qualities = {
        "q(R)": {"form": "array", "loss": "radial fft2", "regularize": "self", "regularize scale": 1.},
        "PA(R)": {"form": "array", "loss": "radial fft2", "regularize": "const", "regularize scale": 0.6},
    }

    fft_start = 10

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)

    def build_parameters(self):
        super().build_parameters()
        for p in self.parameter_specs:
            if "(R)" not in p:
                continue
            if isinstance(self.parameter_specs[p], dict):
                self.parameters[p] = Parameter_Array(p, **self.parameter_specs[p])
            elif isinstance(self.parameter_specs[p], Parameter_Array):
                self.parameters[p] = self.parameter_specs[p]
            else:
                raise ValueError(f"unrecognized parameter specification for {p}")
        
    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["PA(R)"].value is not None:
            self["PA(R)"].set_value(self["PA(R)"].get_values() * np.pi / 180, override_fixed = True)

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        # # Get the subsection of the full image
        # target_area = target[self.window]
        # icenter = coord_to_index(self["center"][0].value, self["center"][1].value, target_area)
        # # Transform the target image area to remove global PA and ellipticity
        # XX, YY = target_area.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        # XX, YY = super().transform_coordinates(XX, YY)
        # Y, X = coord_to_index(XX + self["center"][0].value, YY + self["center"][1].value, target_area)
        # target_transformed = nearest_neighbor(target_area.data, X, Y)
        # Initialize the PA(R) values
        if self["PA(R)"].value is None:
            # iso_info = isophotes(
            #     target_transformed,
            #     (icenter[1], icenter[0]),
            #     pa = 0., q = 1., R = self.profR[1:],
            # )
            self["PA(R)"].set_value([self["PA"].value] + list(self["PA"].value for io in iso_info), override_fixed = True) # (-io['phase2']/2) % np.pi
            
        # Initialize the q(R) values
        if self["q(R)"].value is None:
            # q_R = [1. - 1e-7]
            # q_samples = np.linspace(0.3,0.9,10)
            # for r in self.profR[1:]:
            #     iso_info = isophotes(
            #         target_transformed,
            #         (icenter[1], icenter[0]),
            #         pa = 0., q = q_samples, R = r,
            #     )
            #     q_R.append(0.9)
            self["q(R)"].set_value(np.ones(len(self.profR))*0.9, override_fixed = True)
            
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.window.shape/2)**2)):
                self.profR.append(max(1,self.profR[-1]*1.2))
            self.profR.pop()
            self.profR = np.array(self.profR)

    def transform_coordinates(self, X, Y, R = None, transmit = True):
        if transmit:
            X, Y = super().transform_coordinates(X, Y)

        if R is None:
            R = self.radius_metric(X, Y)
        PA = UnivariateSpline(self.profR, np.unwrap(self["PA(R)"].get_values()*2)/2, ext = "const", s = 0)
        q = UnivariateSpline(self.profR, self["q(R)"].get_values(), ext = "const", s = 0)
        
        return Axis_Ratio_Cartesian(q(R), X, Y, PA(R))

    def _regularize_loss(self):

        params = self.get_parameters(quality = ["regularize", "const"])
        params.update(self.get_parameters(quality = ["regularize", "self"]))
        regularization = np.ones(len(self.profR))
        for P in params:
            if not isinstance(params[P], Parameter_Array):
                continue
            vals = params[P].get_values()
            if params[P].cyclic:
                period_factor = 2*np.pi / (params[P].limits[1] - params[P].limits[0])
                vals = np.unwrap(vals * period_factor) / period_factor
            if self.parameter_qualities[P]["regularize"] == "const":
                reg_scale = np.ones(len(self.profR))
            elif self.parameter_qualities[P]["regularize"] == "self":
                reg_scale = vals
            reg_scale *= self.parameter_qualities[P]["regularize scale"]
            reg = [2*np.abs(vals[1] - vals[0]) / reg_scale[0]]
            for i in range(1, len(vals) - 1):
                reg.append((np.abs(vals[i] - vals[i-1]) + np.abs(vals[i] - vals[i+1])) / reg_scale[i])
            reg.append(2*np.abs(vals[-2] - vals[-1]) / reg_scale[-1])
            regularization += 0.1*np.array(reg)

        return regularization
        
    def compute_loss(self, data):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(data)
        
        if not any(m in self.loss_mode for m in ["default", "radial"]):
            return

        X, Y = data.loss_image.get_coordinate_meshgrid(self["center"][0].value, self["center"][1].value)
        if self.loss_speed_factor != 1:
            X = X[::self.loss_speed_factor,::self.loss_speed_factor]
            Y = Y[::self.loss_speed_factor,::self.loss_speed_factor]
        X, Y = super().transform_coordinates(X, Y)
        preR = self.radius_metric(X, Y)
        preTheta = np.arctan2(Y, X)
        reg = self._regularize_loss()
        rad_bins = [self.profR[0]] + list((self.profR[:-1] + self.profR[1:])/2) + [self.profR[-1]*100]
            
        temp_fft2 = []
        theta_bins = np.linspace(-np.pi, np.pi, 17)
        segment_stats = binned_statistic_2d(preR.ravel(), preTheta.ravel(), data.loss_image.data.ravel(), statistic = 'median', bins = [rad_bins, theta_bins])[0]
        cbins = (theta_bins[:-1] + theta_bins[1:]) / 2 + np.pi
        for i in range(len(self.profR)):
            if self.profR[i] < self.fft_start:
                temp_fft2.append(temp_loss[i])
            else:
                smooth = segment_stats[i]
                N = np.isfinite(smooth)
                if not np.all(N):
                    smooth[np.logical_not(N)] = np.interp(cbins[np.logical_not(N)], cbins[N], smooth[N], period = 2*np.pi)
                coefs = fft(smooth)
                temp_fft2.append(np.abs(coefs[2]) * reg[i])
                
        self.loss["radial fft2"] = np.array(temp_fft2)
