import torch
import numpy as np
import io
from autoprof import plots
import matplotlib.pyplot as plt
from autoprof.utils.conversions.optimization import cyclic_difference_np
from autoprof.utils.conversions.dict_to_hdf5 import dict_to_hdf5
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

    dtype = torch.float32
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

    def compute_loss(self):
        """Compute a standard Chi^2 loss given the target image, model, and
        variance image. Typically if overloaded this will also be
        called with super() and higher methods will multiply or add to
        the loss.

        """
        pixels = torch.prod(self.model_image.window.shape/self.target.pixelscale)
        if self.target.has_mask:
            mask = torch.logical_not(self.target[self.model_image.window].mask)
            pixels = torch.sum(mask)
            self.loss = torch.sum((
                torch.pow(
                    (
                        self.target[self.model_image.window].data
                        - self.model_image.data
                    ),
                    2,
                )
                / self.target[self.model_image.window].variance
            )[mask])
        else:
            self.loss = torch.sum(
                torch.pow((self.target[self.model_image.window] - self.model_image).data, 2)
                / self.target[self.model_image.window].variance
            )
        self.loss /= pixels - np.sum(self.parameter_vector_len)
        if self.constraints is not None:
            for constraint in self.constraints:
                self.loss *= 1 + self.constraint_strength * constraint(self)
        return self.loss

    def step(self, parameters = None, parameters_as_representation = True):
        """Call after updating any of the model parameters.

        """
        self.is_sampled = False
        if parameters is None:
            return
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
        self.step(parameters)
        self.sample()
        self.compute_loss()
        return self.model_image.data

    def full_loss(self, parameters = None):
        self.step(parameters)
        self.sample()
        self.compute_loss()
        return self.loss

    def jacobian(self, parameters):
        return jacobian(
            self.full_sample,
            parameters,
            strategy = "forward-mode",
            vectorize = True,
            create_graph = False,
        )

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
        
