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
        "q(R)": {"units": "b/a", "limits": (0,1), "uncertainty": 0.03},
        "PA(R)": {"units": "rad", "limits": (0,np.pi), "cyclic": True, "uncertainty": 0.06},
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
        XX, YY = Rotate_Cartesian(self["PA"].value, XX, YY)
        YY /= self["q"].value
        Y, X = coord_to_index(XX + self["center_x"].value, YY + self["center_y"].value, target_area)
        target_transformed = nearest_neighbor(target_area.data, X, Y)
        # Initialize the PA(R) values
        if self["PA(R)"].value is None:
            iso_info = isophotes(
                target_transformed,
                (icenter[1], icenter[0]),
                pa = 0., q = 1., R = self.profR[1:],
            )
            self["PA(R)"].set_value([1e-7] + list((-io['phase2']/2) % np.pi for io in iso_info), override_fixed = True)
            # First point fixed PA since no orientation meaningful at R = 0
            self["PA(R)"].value[0].user_fixed = True
            self["PA(R)"].value[0].update_fixed(True)
        plt.imshow(
            np.log10(target_transformed),
            origin="lower",
            # norm=ImageNormalize(stretch=HistEqStretch(target_transformed), clip = False),
        )
        plt.plot((self.profR*np.cos(self["PA(R)"].get_values()) + self["center_x"].value - target_area.origin[1])/0.262, (self.profR*np.sin(self["PA(R)"].get_values()) + self["center_y"].value - target_area.origin[0])/0.262, color = 'r')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("stretched_target.jpg")
        plt.close()
            
        # Initialize the q(R) values
        if self["q(R)"].value is None:
            q_R = [1. - 1e-7]
            q_samples = np.linspace(0.3,0.7,10)
            for r in self.profR[1:-1]:
                iso_info = isophotes(
                    target_transformed,
                    (icenter[1], icenter[0]),
                    pa = 0., q = q_samples, R = r,
                )
                q_R.append(q_samples[np.argmin(list(iso["amplitude2"] for iso in iso_info))])
            q_R.append(1. - 1e-7)
            self["q(R)"].set_value(q_R, override_fixed = True)
            # First and last points required to be circular for integration with global PA/q
            self["q(R)"].value[0].user_fixed = True
            self["q(R)"].value[0].update_fixed(True)
            self["q(R)"].value[-1].user_fixed = True        
            self["q(R)"].value[-1].update_fixed(True)
            
    def set_window(self, *args, **kwargs):
        super().set_window(*args, **kwargs)

        if self.profR is None:
            self.profR = [0,1]
            while self.profR[-1] < np.sqrt(np.sum((self.window.shape/2)**2)):
                self.profR.append(self.profR[-1]*1.2)
            self.profR.pop()
            self.profR = np.array(self.profR)
    
    def sample_model(self, sample_image = None, X = None, Y = None, R = None):
        if sample_image is None:
            sample_image = self.model_image

        if X is None or Y is None:
            X, Y = sample_image.get_coordinate_meshgrid(self["center_x"].value, self["center_y"].value)
        if R is None:
            R = np.sqrt(X**2 + Y**2)
            
        X, Y = Rotate_Cartesian(-self["PA"].value, X, Y)
        Y /= self["q"].value
        PA = UnivariateSpline(self.profR, self["PA(R)"].get_values(), ext = "const", s = 0)
        q = UnivariateSpline(self.profR, self["q(R)"].get_values(), ext = "const", s = 0)
        X, Y = Rotate_Cartesian(-PA(R), X, Y)
        Y /= q(R)
        
        super().sample_model(sample_image, X = X, Y = Y, R = np.sqrt(X**2 + Y**2))
                
