from .galaxy_model_object import Galaxy_Model
from .parameter_object import Parameter_Array
import numpy as np
from scipy.interpolate import UnivariateSpline
from autoprof.utils.initialize import isophotes
from autoprof.utils.interpolate import nearest_neighbor
from autoprof.utils.angle_operations import Angle_Average
from autoprof.utils.conversions.coordinates import Rotate_Cartesian, coord_to_index, index_to_coord
from scipy.stats import iqr
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch, LogStretch, HistEqStretch
from astropy.visualization.mpl_normalize import ImageNormalize

class Warp_Galaxy(Galaxy_Model):

    model_type = " ".join(("warp", Galaxy_Model.model_type))
    parameter_specs = {
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.04},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.1},
    }
    parameter_qualities = {
        "q(R)": {"loss": "local radial"},
        "PA(R)": {"loss": "local radial"},
    }

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
            for i in range(len(self["PA(R)"])):
                self["PA(R)"].set_value(self["PA(R)"][i].value * np.pi / 180, override_fixed = True, index = i)

    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        if not (self["PA(R)"].value is None or self["q(R)"].value is None):
            return

        # Get the subsection of the full image
        target_area = target[self.window]
        icenter = coord_to_index(self["center_x"].value, self["center_y"].value, target_area)
        # Transform the target image area to remove global PA and ellipticity
        XX, YY = target_area.get_coordinate_meshgrid(self["center_x"].value, self["center_y"].value)
        XX, YY = Rotate_Cartesian(-self["PA"].value, XX, YY)
        YY /= self["q"].value
        XX, YY = Rotate_Cartesian(self["PA"].value, XX, YY)
        Y, X = coord_to_index(XX + self["center_x"].value, YY + self["center_y"].value, target_area)
        target_transformed = nearest_neighbor(target_area.data, X, Y)
        # Initialize the PA(R) values
        if self["PA(R)"].value is None:
            iso_info = isophotes(
                target_transformed,
                (icenter[1], icenter[0]),
                pa = 0., q = 1., R = self.profR[1:],
            )
            self["PA(R)"].set_value([1e-7] + list(self["PA"].value for io in iso_info), override_fixed = True) # (-io['phase2']/2) % np.pi
            # First point fixed PA since no orientation meaningful at R = 0
            self["PA(R)"].value[0].user_fixed = True
            self["PA(R)"].value[0].update_fixed(True)
            
        # Initialize the q(R) values
        if self["q(R)"].value is None:
            q_R = [1. - 1e-7]
            q_samples = np.linspace(0.3,0.9,10)
            for r in self.profR[1:]:
                iso_info = isophotes(
                    target_transformed,
                    (icenter[1], icenter[0]),
                    pa = 0., q = q_samples, R = r,
                )
                q_R.append(0.8)
                #q_R.append(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))])
            self["q(R)"].set_value(q_R, override_fixed = True)
            # First point required to be circular since no shape at R = 0
            self["q(R)"].value[0].user_fixed = True
            self["q(R)"].value[0].update_fixed(True)
            
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.window.shape/2)**2)):
                self.profR.append(self.profR[-1]*1.2)
            self.profR.pop()
            self.profR = np.array(self.profR)
    
    def sample_model(self, sample_image = None, X = None, Y = None):
        if sample_image is None:
            sample_image = self.model_image

        if X is None or Y is None:
            X, Y = sample_image.get_coordinate_meshgrid(self["center_x"].value, self["center_y"].value)

        sample_image, X, Y = super().sample_model(sample_image, X = X, Y = Y)

        R = self.radius_metric(X, Y)
        PA = UnivariateSpline(self.profR, np.unwrap(self["PA(R)"].get_values()*2)/2, ext = "const", s = 0)
        q = UnivariateSpline(self.profR, self["q(R)"].get_values(), ext = "const", s = 0)
        X, Y = Rotate_Cartesian(-PA(R), X, Y)
        Y /= q(R)

        return sample_image, X, Y
                        
    def compute_loss(self, loss_image):
        # If the image is locked, no need to compute the loss
        if self.locked:
            return

        super().compute_loss(loss_image)

        loss_image, X, Y = super().sample_model(loss_image)

        R = self.radius_metric(X, Y)
        temp_loss = [np.mean(loss_image.data[R <= self.profR[1]])]
        for i in range(1, len(self.profR)-1):
            temp_loss.append(np.mean(loss_image.data[np.logical_and(R >= self.profR[i-1], R <= self.profR[i+1])]))
        temp_loss.append(np.mean(loss_image.data[R >= self.profR[-2]]))
        self.loss["local radial"] = np.array(temp_loss)

    def get_loss_history(self, limit = np.inf):
        
        super().get_loss_history(limit)
        param_order = self.get_parameters(exclude_fixed = True, quality = ["loss", "local radial"]).keys()
        for ir in range(len(self.profR)):
            params = []
            loss_history = []
            for i in range(min(limit, len(self.loss_history))):
                params_i = self.get_parameters(index = i if i > 0 else None, exclude_fixed = True, quality = ["loss", "local radial"])
                sub_params = []
                for P in param_order:
                    if isinstance(params_i[P], Parameter_Array):
                        sub_params.append(params_i[P][ir])
                    elif isinstance(params_i[P], Parameter):
                        sub_params.append(params_i[P])
                params.append(np.array(sub_params))
                loss_history.append(self.get_loss(i, loss_quality = "local radial")[ir])
            yield loss_history, params
        
