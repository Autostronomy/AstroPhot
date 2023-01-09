import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from ..utils.conversions.optimization import cyclic_difference_np
from ..utils.conversions.dict_to_hdf5 import dict_to_hdf5
from ..utils.optimization import reduced_chi_squared
from ..image import Model_Image, Window, Target_Image
from .parameter_object import Parameter
from copy import copy
from time import time
from .. import AP_config
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
    constraint_strength = 10.
    
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

    def __init__(self, name, *args, target = None, window = None, locked = False, **kwargs):
        assert ":" not in name and "|" not in name, "characters '|' and ':' are reserved for internal model operations please do not include these in a model name"
        self.name = name
        self.constraints = kwargs.get("constraints", None)
        self.equality_constraints = []
        self.requires_grad = kwargs.get("requires_grad", False)
        self.target = target
        self.window = window
        self._locked = locked

    def add_equality_constraint(self, model, parameter):
        if isinstance(parameter, (tuple, list)):
            for P in parameter:
                self.add_equality_constraint(model, P)
            return
        self.parameters[parameter] = model[parameter]
        self.equality_constraints.append(parameter)
        model.equality_constraints.append(parameter)
        
    @torch.no_grad()
    def initialize(self, target = None):
        """When this function finishes, all parameters should have numerical
        values (non None) that are reasonable estimates of the final
        values.

        """
        pass

    def make_model_image(self):
        return Model_Image(
            window = self.window,
            pixelscale = self.target.pixelscale,
        )
        
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
            np.sum(self.parameter_vector_len()),
            self.target[self.window].mask,
            self.target[self.window].variance
        )
        if self.constraints is not None:
            for constraint in self.constraints:
                loss *= 1 + self.constraint_strength * constraint(self)
        if return_sample:
            return loss, model_image
        return loss

    def set_parameters(self, parameters, override_locked = False, as_representation = True, parameters_identity = None):
        """
        Set the parameter values for this model with a given object.

        Parameters:
            parameters: updated values for the parameters. Either as a dictionary of parameter_name: tensor pairs, or as a 1D tensor.
            override_locked: locked parameters normally are ignored, set this to True to include locked parameters
            as_representation: if true the parameters are given as a representation form, if false then the parameters are given as values (see parameters for difference between representation and value)
            parameters_identity: iterable of parameter names if "parameters" is some subset of the full parameter tensor. Note that parameters_identity must be in the same order as model.parameter_order, though any combination can be included/missing
        """
        if isinstance(parameters, dict):
            for P in parameters:
                if not override_locked and self[P].locked:
                    continue
                if as_representation:
                    self[P].representation = parameters[P]
                else:
                    self[P].value = parameters[P]
            return
        parameters = torch.as_tensor(parameters, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        start = 0
        for P, V in zip(self.parameter_order(override_locked = override_locked), self.parameter_vector_len(override_locked = override_locked)):
            if parameters_identity is None or P in parameters_identity:
                if as_representation:
                    self[P].representation = parameters[start:start + V].reshape(self[P].representation.shape)
                else:
                    self[P].value = parameters[start:start + V].reshape(self[P].value.shape)
            start += V
        
    def set_uncertainty(self, uncertainty, override_locked = False, as_representation = False):
        if isinstance(uncertainty, dict):
            for P in uncertainty:
                if not override_locked and self[P].locked:
                    continue
                self[P].set_uncertainty(
                    uncertainty[P],
                    override_locked = override_locked,
                    as_representation = as_representation
                )
            return
        uncertainty = torch.as_tensor(uncertainty, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        start = 0
        for P, V in zip(self.parameter_order(override_locked = override_locked), self.parameter_vector_len(override_locked = override_locked)):
            self[P].set_uncertainty(
                uncertainty[start:start + V].reshape(self[P].representation.shape),
                override_locked = override_locked,
                as_representation = as_representation,
            )
            start += V        
        
    def full_sample(self, parameters = None, as_representation = True, override_locked = False, return_data = True, flatten = False, parameters_identity = None):
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation, parameters_identity = parameters_identity)
        if flatten:
            return self.sample().flatten()
        if return_data:
            return self.sample().data
        return self.sample()

    def full_loss(self, parameters = None, as_representation = False, override_locked = False):
        if parameters is not None:
            self.set_parameters(parameters, as_representation = as_representation, override_locked = override_locked)
        return self.compute_loss()

    def jacobian(self, parameters = None, as_representation = False, override_locked = False, flatten = False):
        raise NotImplementedError

    @property
    def window(self):
        try:
            if self._window is None:
                return self.target.window.make_copy()
            return self._window
        except AttributeError:
            return self.target.window.make_copy()

    def set_window(self, window):
        # If no window given, dont go any further
        if window is None:
            return
    
        # If the window is given in proper format, simply use as-is
        if isinstance(window, Window):
            self._window = window
        elif len(window) == 2:
            self._window = Window(
                origin = self.target.origin + torch.tensor((window[0][0],window[1][0]), dtype = AP_config.ap_dtype, device = AP_config.ap_device)*self.target.pixelscale,
                shape = torch.tensor((window[0][1] - window[0][0], window[1][1] - window[1][0]), dtype = AP_config.ap_dtype, device = AP_config.ap_device)*self.target.pixelscale,
            )
        elif len(window) == 4:
            self._window = Window(
                origin = torch.tensor((window[0],window[2]), dtype = AP_config.ap_dtype, device = AP_config.ap_device),
                shape = torch.tensor((window[1] - window[0], window[3] - window[2]), dtype = AP_config.ap_dtype, device = AP_config.ap_device),
            )
        else:
            raise ValueError(f"Unrecognized window format: {str(window)}")
            
    @window.setter
    def window(self, window):
        self._window = window
        self.set_window(window)

    @property 
    def target(self):
        try:
            if self._target is None:
                raise RuntimeError(f"Model {self.name} currently has no target, set a target image for this model.")
            return self._target
        except AttributeError:
            raise RuntimeError(f"Model {self.name} currently has no target, set a target image for this model.")
    @target.setter
    def target(self, tar):
        self._target = tar
        assert tar is None or isinstance(tar, Target_Image)
        
    @property
    def locked(self):
        """Set when the model should remain fixed going forward. This model
        will be bypassed when fitting parameters, however it will
        still be sampled for generating the model image.

        """
        return self._locked
    @locked.setter
    def locked(self, val):
        assert isinstance(val, bool)
        self._locked = val
    @property
    def requires_grad(self):
        return self._requires_grad
    @requires_grad.setter
    def requires_grad(self, val):
        assert isinstance(val, bool)
        self._requires_grad = val
        for P in self.parameters:
            self[P].requires_grad = val
            
    def parameter_vector_len(self, override_locked = False):
        return list(int(np.prod(self[P].value.shape)) for P in self.parameter_order(override_locked = override_locked))
    def get_parameter_vector(self, as_representation = False, override_locked = False):
        parameters = torch.zeros(np.sum(self.parameter_vector_len(override_locked = override_locked)), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        vstart = 0
        for P, V in zip(
                self.parameter_order(override_locked = override_locked),
                self.parameter_vector_len(override_locked = override_locked)
        ):
            if as_representation:
                parameters[vstart: vstart + V] = self[P].representation
            else:
                parameters[vstart: vstart + V] = self[P].value
            vstart += V
        return parameters
                
    def __str__(self):
        """String representation for the model."""
        return str(self.get_state())

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
        
    def __eq__(self, other):
        return self is other
