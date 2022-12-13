import torch
import numpy as np
import io
from autoprof import plots
import matplotlib.pyplot as plt
from autoprof.utils.conversions.optimization import cyclic_difference_np
from autoprof.utils.conversions.dict_to_hdf5 import dict_to_hdf5
from autoprof.utils.optimization import reduced_chi_squared
from autoprof.image import Model_Image, AP_Window
from copy import copy
from time import time
from torch.autograd.functional import jacobian
__all__ = ["AutoProf_Model"]

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

class AutoProf_Model(object):
    """AutoProf_Model(name, *args, filename = None, model_type = None, **kwargs)

    Core class for all AutoProf models and model like objects. The
    signature defined for this class includes all expected behaviour
    that will be accessed by some or all optimizers during
    fitting. This base class also handles saving and loading of
    models, though individual models should define thier "get_state"
    behaviour and "load" behaviour to fully take advantage of this
    functionality.

    Parameters:
        name: every AutoProf model should have a unique name [str]
        filename: name of a file to load AutoProf parameters, window, and name. The model will still need to be told its target, device, and other information [str]
        model_type: a model type string can determine which kind of AutoProf model is instantiated [str]
    """

    model_type = ""

    dtype = torch.float64
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __new__(cls, *args, filename = None, model_type = None, **kwargs):
        if filename is not None:
            state = AutoProf_Model.load(filename)
            MODELS = all_subclasses(AutoProf_Model)
            for M in MODELS:
                if M.model_type == state["model_type"]:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(f"Unknown AutoProf model type: {state['model_type']}")
        elif model_type is not None:
            MODELS = all_subclasses(AutoProf_Model)
            for M in MODELS:
                if M.model_type == model_type:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(f"Unknown AutoProf model type: {model_type}")
            
        return super().__new__(cls)

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.epoch = None
        self.constraints = kwargs.get("constraints", None)
        self.is_sampled = False
        
    def initialize(self):
        """When this function finishes, all parameters should have numerical
        values (non None) that are reasonable estimates of the final
        values.

        """
        pass

    def make_model_image(self):
        return Model_Image(
            window = self.window,
            pixelscale = self.target.pixelscale,
            dtype = self.dtype,
            device = self.device,
        )
        
    def startup(self):
        """Run immediately before fitting begins. Typically this is not
        needed, though it is available for models which require
        specific configuration for fitting, see also "finalize"

        """
        self.parameter_vector_len = list(int(np.prod(self[P].value.shape)) for P in self.parameter_order)
        
    def finalize(self):
        """This is run after fitting and can be used to undo any fitting
        specific settings or aspects of a model.

        """
        pass

    def sample(self, sample_image=None):
        """Calling this function should fill the given image with values
        sampled from the given model.

        """
        pass

    def compute_loss(self, return_sample = False):
        """Compute a standard Chi^2 loss given the target image, model, and
        variance image. Typically if overloaded this will also be
        called with super() and higher methods will multiply or add to
        the loss.

        """
        model_image = self.sample()
        loss = reduced_chi_squared(
            self.target[self.window].data,
            model_image.data,
            np.sum(self.parameter_vector_len),
            self.target[self.window].mask,
            self.target[self.window].variance
        )
        # if self.constraints is not None:
        #     for constraint in self.constraints:
        #         loss *= 1 + self.constraint_strength * constraint(self)
        if return_sample:
            return loss, model_image
        return loss

    def set_parameters(self, parameters, override_locked = False, parameters_as_representation = True):
        if isinstance(parameters, dict):
            for P in parameters:
                isinstance(parameters[P], torch.Tensor)
                if parameters_as_representation:
                    self[P].representation = parameters[P]
                else:
                    self[P].value = parameters[P]
            return
        parameters = torch.as_tensor(parameters, dtype = self.dtype, device = self.device)
        start = 0
        for P, V in zip(self.parameter_order, self.parameter_vector_len):
            if parameters_as_representation:
                self[P].representation = parameters[start:start + V].reshape(self[P].representation.shape)
            else:
                self[P].value = parameters[start:start + V].reshape(self[P].value.shape)
            start += V
        
    def set_uncertainty(self, uncertainty, override_locked = False, uncertainty_as_representation = False):
        uncertainty = torch.as_tensor(uncertainty, dtype = self.dtype, device = self.device)
        start = 0
        for P, V in zip(self.parameter_order, self.parameter_vector_len):
            self[P].set_uncertainty(
                uncertainty[start:start + V].reshape(self[P].representation.shape),
                uncertainty_as_representation = uncertainty_as_representation,
            )
            start += V        
        
    def get_parameters_representation(self):
        """Get the optimizer friently representations of all the non-locked
        parameter values for this model.

        """
        pass

    def get_parameters_value(self):
        """Get the non-locked parameter values for this model."""
        pass

    def full_sample(self, parameters = None):
        if parameters is not None:
            self.set_parameters(parameters)
        return self.sample().data

    def full_loss(self, parameters = None):
        if parameters is not None:
            self.set_parameters(parameters)
        return self.compute_loss()

    def jacobian(self, parameters):
        return jacobian(
            self.full_sample,
            parameters,
            strategy = "forward-mode",
            vectorize = True,
            create_graph = False,
        )

    @property
    def window(self): # fixme allow None window that just reproduces full image
        return self._window
    def set_window(self, window):
        # If no window given, use the whole image
        if window is None:
            window = [
                [0, self.target.data.shape[1]],
                [0, self.target.data.shape[0]],
            ]
            index_units = False
    
        # If the window is given in proper format, simply use as-is
        if isinstance(window, AP_Window):
            self._window = window
        else:
            self._window = AP_Window(
                origin = self.target.origin + torch.tensor((window[0][0],window[1][0]), dtype = self.dtype, device = self.device)*self.target.pixelscale,
                shape = torch.tensor((window[0][1] - window[0][0], window[1][1] - window[1][0]), dtype = self.dtype, device = self.device)*self.target.pixelscale,
                dtype = self.dtype,
                device = self.device,
            )
        if self._base_window is None:
            self._base_window = self._window.make_copy()
    
        self._window = self._window & self.target.window
    
    @window.setter
    def window(self, window):
        self.set_window(window)
        self._window.to(dtype = self.dtype, device = self.device)

    def __str__(self):
        """String representation for the model."""
        return "AutoProf Model Instance"

    def get_state(self):
        state = {
            "name": self.name,
            "model_type": self.model_type,
        }
        return state

    def save(self, filename = "AutoProf.yaml"):
        if filename.endswith(".yaml"):
            import yaml
            state = self.get_state()
            with open(filename, "w") as f:
                yaml.dump(state, f, indent = 2) 
        elif filename.endswith(".json"):
            import json
            state = self.get_state()
            with open(filename, "w") as f:
                json.dump(state, f, indent = 2)
        elif filename.endswith(".hdf5"):
            import h5py
            state = self.get_state()
            with h5py.File(filename, "w") as F:
                dict_to_hdf5(F, state)
        else:
            if isinstance(filename, str) and '.' in filename:
                raise ValueError(f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5")
            else:
                raise ValueError(f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5")

    @classmethod
    def load(cls, filename = "AutoProf.yaml"):
        if isinstance(filename, dict):
            state = filename
        elif isinstance(filename, io.TextIOBase):
            import yaml
            state = yaml.load(filename, Loader = yaml.FullLoader)            
        elif filename.endswith(".yaml"):
            import yaml
            with open(filename, "r") as f:
                state = yaml.load(f, Loader = yaml.FullLoader)            
        elif filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                state = json.load(f)
        elif filename.endswith(".hdf5"):
            import h5py
            with h5py.File(filename, "r") as F:
                state = hdf5_to_dict(F)
        else:
            if isinstance(filename, str) and '.' in filename:
                raise ValueError(f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5")
            else:
                raise ValueError(f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5 or python dictionary.")
        return state
        
