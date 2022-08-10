from .galaxy_model_object import Galaxy_Model
from .warp_model import Warp_Galaxy
from .parameter_object import Parameter_Array
from autoprof.utils.initialize import isophotes
from autoprof.utils.isophote.ellipse import parametric_SuperEllipse
from autoprof.utils.parametric_profiles import sersic
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord, Axis_Ratio_Cartesian
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import binned_statistic, iqr, binned_statistic_2d
from scipy.ndimage import distance_transform_edt
from copy import copy
from functools import reduce

class Isophote_Galaxy(Galaxy_Model):

    model_type = " ".join(("isophote", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
    }
    parameter_qualities = {
        "q(R)": {"form": "array", "regularize": "const", "regularize scale": 1.},
        "PA(R)": {"form": "array", "regularize": "const", "regularize scale": 1.},
    }

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "profR"):
            self.profR = None
        super().__init__(*args, **kwargs)
        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < (max(self.window.shape)/2):
                self.profR.append(self.profR[-1] + max(1,self.profR[-1]*0.2))
            self.profR.pop()
            self.profR = np.array(self.profR)
            
    def _init_convert_input_units(self):
        super()._init_convert_input_units()
        
        if self["PA(R)"].value is not None:
            self["PA(R)"].set_value(self["PA(R)"].get_value() * np.pi / 180, override_fixed = True)

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        self["PA"].update_fixed(True)
        self["q"].update_fixed(True)
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        if self["PA(R)"].value is None:
            self["PA(R)"].set_value(np.ones(len(self.profR))*self["PA"].value, override_fixed = True)
            
        if self["q(R)"].value is None:
            self["q(R)"].set_value(np.ones(len(self.profR))*self["q"].value, override_fixed = True)
            
    def evaluate_model(self, X, Y, image):
        X, Y = self.transform_coordinates(X, Y)
        
        R = self.radius_metric(X, Y)
        
        PA = UnivariateSpline(self.profR, np.unwrap(self["PA(R)"].get_value()*2)/2, ext = "const", s = 0)
        q = UnivariateSpline(self.profR, self["q(R)"].get_value(), ext = "const", s = 0)
        
        X, Y = Axis_Ratio_Cartesian(q(R), X, Y, PA(R), inv_scale = False)

        Z = binned_statistic_2d(X.ravel(), Y.ravel(), self.radial_model(R, image).ravel(), statistic = "mean", bins = R.shape)[0]

        M = np.logical_not(np.isfinite(Z))
        nearest_neighbor = distance_transform_edt(M, return_distances=False, return_indices=True)

        Z[M] = Z[nearest_neighbor[0][M], nearest_neighbor[1][M]]
        Z[self.radius_metric(X, Y) >= self.profR[-1]] = 0

        return Z
        
    def regularize_loss(self):

        params = {"q(R)": self["q(R)"], "PA(R)": self["PA(R)"]}
        regularization = 0. #np.ones(len(self.profR)-1)
        n_reg = 0
        rscale = (self.profR[1:] + self.profR[:-1])**2
        rscale /= rscale[-1]
        for P in params:
            if not isinstance(params[P], Parameter_Array):
                continue
            vals = params[P].get_representation()
            if params[P].cyclic:
                period_factor = 2*np.pi / (params[P].limits[1] - params[P].limits[0])
                vals = np.unwrap(vals * period_factor) / period_factor
            if self.parameter_qualities[P]["regularize"] == "const":
                reg_scale = np.ones(len(self.profR)-1)
            elif self.parameter_qualities[P]["regularize"] == "self":
                reg_scale = (vals[1:] + vals[:-1])/2
            reg_scale *= self.parameter_qualities[P]["regularize scale"]
            reg = []
            for i in range(len(vals) - 1):
                reg.append(np.abs(vals[i] - vals[i+1]) * rscale[i] / reg_scale[i])
            regularization += 0.00001*np.sum(reg)
            n_reg += len(reg)
        return 1 + regularization/n_reg

    def finalize(self):
        dat = self.target[self.window]
        dat.data -= np.median(dat.data)
        noise = 0.5*iqr(dat.data,rng=[16,84])/2
        plt.figure(figsize=(6, 6))
        plt.imshow(
            dat.data,
            origin="lower",
            cmap="Greys",
            norm=ImageNormalize(stretch=HistEqStretch(dat.data[dat.data <= 3*noise]), clip = False, vmax = 3*noise, vmin = np.min(dat.data)),
        )
        my_cmap = copy(cm.Greys_r)
        my_cmap.set_under("k", alpha=0)
    
        plt.imshow(
            np.ma.masked_where(dat.data < 3*noise, dat.data), 
            origin="lower",
            cmap=my_cmap,
            norm=ImageNormalize(stretch=LogStretch(),clip = False),
            clim=[3 * noise, None],
            interpolation = 'none',
        )
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
        plt.xlim([0, dat.data.shape[1]])
        plt.ylim([0, dat.data.shape[0]])

        icenter = coord_to_index(self["center"][0].value, self["center"][1].value, dat)
        for i in range(len(self.profR)):
            N = max(15, int(0.9 * 2 * np.pi * self.profR[i] / dat.pixelscale))
            theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
            theta = np.arctan((self["q(R)"][i].value) * np.tan(theta)) + np.pi * (
                np.cos(theta) < 0
            )
            R = self.profR[i] / dat.pixelscale
            X, Y = parametric_SuperEllipse(
                theta,
                1 - self["q(R)"][i].value,
                2,
            )
            X, Y = Rotate_Cartesian(-self["PA(R)"][i].value, X, Y)
            X, Y = Axis_Ratio_Cartesian(self["q"].value, X, Y, -self["PA"].value, inv_scale = True)
            rescale = np.max(np.sqrt(X**2 + Y**2))
            X, Y = (
                R * X / rescale + icenter[1],
                R * Y / rescale + icenter[0],
            )
        
            plt.plot(
                list(X) + [X[0]],
                list(Y) + [Y[0]],
                linewidth=((i + 1) / len(self.profR)) ** 2,
                color="r",
            )
        plt.savefig(
            f"isophotes_{self.name}.jpg",
            dpi=600,
        )
        plt.close()
        
        


# class Isophote_Galaxy(Galaxy_Model):

#     model_type = " ".join(("isophote", Galaxy_Model.model_type))
#     parameter_specs = {
#         "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
#         "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.08},
#         "center": {"units": "arcsec", "fixed": True, "uncertainty": 0.0},
#     }
#     parameter_qualities = {
#         "q(R)": {"form": "array", "loss": "band fft2", "regularize": "self", "regularize scale": 1.},
#         "PA(R)": {"form": "array", "loss": "band fft2", "regularize": "const", "regularize scale": 0.2},
#     }

#     def __init__(self, *args, **kwargs):
#         if not hasattr(self, "profR"):
#             self.profR = None
#         super().__init__(*args, **kwargs)
        
#     def set_window(self, *args, **kwargs):
#         super().set_window(*args, **kwargs)

#         if self.profR is None:
#             self.profR = [0,1]
#             while self.profR[-1] < min(self.window.shape/2):
#                 self.profR.append(max(1,self.profR[-1]*1.2))
#             self.profR.pop()                
#             self.profR = np.array(self.profR)
        
#     def initialize(self, target = None):
#         if target is None:
#             target = self.target
#         super().initialize(target)
        
#         if self["PA(R)"].value is None:
#             self["PA(R)"].set_value(np.ones(len(self.profR))*self["PA"].value, override_fixed = True)
            
#         if self["q(R)"].value is None:
#             self["q(R)"].set_value(np.ones(len(self.profR))*0.9, override_fixed = True)

#         if not hasattr(self, "profI"):
#             self.profI = np.zeros(len(self.profR))            

#     def finalize(self):
#         dat = self.target[self.window]
#         dat.data -= np.median(dat.data)
#         noise = 0.5*iqr(dat.data,rng=[16,84])/2
#         plt.figure(figsize=(6, 6))
#         plt.imshow(
#             dat.data,
#             origin="lower",
#             cmap="Greys",
#             norm=ImageNormalize(stretch=HistEqStretch(dat.data[dat.data <= 3*noise]), clip = False, vmax = 3*noise, vmin = np.min(dat.data)),
#         )
#         my_cmap = copy(cm.Greys_r)
#         my_cmap.set_under("k", alpha=0)
    
#         plt.imshow(
#             np.ma.masked_where(dat.data < 3*noise, dat.data), 
#             origin="lower",
#             cmap=my_cmap,
#             norm=ImageNormalize(stretch=LogStretch(),clip = False),
#             clim=[3 * noise, None],
#             interpolation = 'none',
#         )
#         plt.xticks([])
#         plt.yticks([])
#         plt.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05)
#         plt.xlim([0, dat.data.shape[1]])
#         plt.ylim([0, dat.data.shape[0]])

#         icenter = coord_to_index(self["center"][0].value, self["center"][1].value, dat)
#         for i in range(len(self.profR)):
#             N = max(15, int(0.9 * 2 * np.pi * self.profR[i] / dat.pixelscale))
#             theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
#             theta = np.arctan((self["q(R)"][i].value) * np.tan(theta)) + np.pi * (
#                 np.cos(theta) < 0
#             )
#             R = self.profR[i] / dat.pixelscale
#             X, Y = parametric_SuperEllipse(
#                 theta,
#                 1 - self["q(R)"][i].value,
#                 2,
#             )
#             X, Y = Rotate_Cartesian(self["PA(R)"][i].value, X, Y)
#             X, Y = (
#                 R * X + icenter[1],
#                 R * Y + icenter[0],
#             )
        
#             plt.plot(
#                 list(X) + [X[0]],
#                 list(Y) + [Y[0]],
#                 linewidth=((i + 1) / len(self.profR)) ** 2,
#                 color="r",
#             )
#         plt.savefig(
#             f"isophotes_{self.name}.jpg",
#             dpi=600,
#         )
#         plt.close()
        

#     def regularize_loss(self):

#         params = {"q(R)": self["q(R)"], "PA(R)": self["PA(R)"]}
#         regularization = np.ones(len(self.profR))
#         for P in params:
#             if not isinstance(params[P], Parameter_Array):
#                 continue
#             vals = params[P].get_value()
#             if params[P].cyclic:
#                 period_factor = 2*np.pi / (params[P].limits[1] - params[P].limits[0])
#                 vals = np.unwrap(vals * period_factor) / period_factor
#             if self.parameter_qualities[P]["regularize"] == "const":
#                 reg_scale = np.ones(len(self.profR))
#             elif self.parameter_qualities[P]["regularize"] == "self":
#                 reg_scale = vals
#             reg_scale *= self.parameter_qualities[P]["regularize scale"]
#             reg = [2*np.abs(vals[1] - vals[0]) / reg_scale[0]]
#             for i in range(1, len(vals) - 1):
#                 reg.append((np.abs(vals[i] - vals[i-1]) + np.abs(vals[i] - vals[i+1])) / reg_scale[i])
#             reg.append(2*np.abs(vals[-2] - vals[-1]) / reg_scale[-1])
#             regularization += 0.01*np.array(reg)

#         return regularization
        
#     def compute_loss(self, data):
#         # If the image is locked, no need to compute the loss
#         if self.locked:
#             return

#         super().compute_loss(data)

#         icenter = coord_to_index(self["center"][0].value, self["center"][1].value, data.residual_image)
        
#         samples = isophotes(data.residual_image.data, icenter, pa = self["PA(R)"].get_value(), q = self["q(R)"].get_value(), R = self.profR / data.residual_image.pixelscale)
#         reg = self.regularize_loss()

#         self.loss["band fft2"] = np.array(list(s["amplitude2"]*r for s, r in zip(samples, reg)))
#         print(self.loss["band fft2"])
#         self.profI = np.array(list(s["flux"] for s in samples))
        
#     def radial_model(self, R, sample_image = None):
#         if sample_image is None:
#             sample_image = self.model_image

#         I = UnivariateSpline(self.profR, self.profI * sample_image.pixelscale**2, ext = "zeros", s = 0)
#         return I(R)
        
#     def sample_model(self, sample_image = None):
#         self.is_sampled = True

        
