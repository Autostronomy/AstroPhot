from copy import copy
from time import time
import io
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from ..utils.conversions.optimization import cyclic_difference_np
from ..utils.conversions.dict_to_hdf5 import dict_to_hdf5
from ..utils.optimization import reduced_chi_squared
from ..image import Model_Image, Window, Target_Image
from .parameter_object import Parameter
from ._shared_methods import select_target, select_sample
from .. import AP_config

__all__ = ["AutoProf_Model"]


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )


######################################################################
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

    model_type = "model"
    constraint_strength = 10.0
    useable = False

    def __new__(cls, *args, filename=None, model_type=None, **kwargs):
        if filename is not None:
            state = AutoProf_Model.load(filename)
            MODELS = AutoProf_Model.List_Models()
            for M in MODELS:
                if M.model_type == state["model_type"]:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(
                    f"Unknown AutoProf model type: {state['model_type']}"
                )
        elif model_type is not None:
            MODELS = AutoProf_Model.List_Models()  # all_subclasses(AutoProf_Model)
            for M in MODELS:
                if M.model_type == model_type:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(f"Unknown AutoProf model type: {model_type}")

        return super().__new__(cls)

    def __init__(self, name, *args, target=None, window=None, locked=False, **kwargs):
        assert (
            ":" not in name and "|" not in name
        ), "characters '|' and ':' are reserved for internal model operations please do not include these in a model name"
        self.name = name
        AP_config.ap_logger.debug("Creating model named: {self.name}")
        self.constraints = kwargs.get("constraints", None)
        self.equality_constraints = []
        self.parameters = {}
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
    @select_target
    def initialize(self, target, *args, **kwargs):
        """When this function finishes, all parameters should have numerical
        values (non None) that are reasonable estimates of the final
        values.

        """
        pass

    def make_model_image(self, window: Optional[Window] = None):
        if window is None:
            window = self.window
        else:
            window = self.window & window
        return self.target[window].model_image()

    def sample(self, image=None, window=None, *args, **kwargs):
        """Calling this function should fill the given image with values
        sampled from the given model.

        """
        pass

    def negative_log_likelihood(
        self,
        parameters=None,
        as_representation=True,
        parameters_identity=None,
    ):
        if parameters is not None:
            self.set_parameters(parameters, as_representation, parameters_identity)

        model = self.sample()
        data = self.target[self.window]
        variance = data.variance
        if self.target.has_mask:
            mask = torch.logical_not(data.mask)
            chi2 = torch.sum(
                ((model - data).data ** 2 / variance)[mask]
            ) / 2.
        else:
            chi2 = torch.sum(
                ((model - data).data ** 2 / variance)
            ) / 2.
            
        return chi2
        

    def set_parameters(
        self,
        parameters,
        as_representation=True,
        parameters_identity=None,
    ):
        """
        Set the parameter values for this model with a given object.

        Parameters:
            parameters: updated values for the parameters. Either as a dictionary of parameter_name: tensor pairs, or as a 1D tensor.
            as_representation: if true the parameters are given as a representation form, if false then the parameters are given as values (see parameters for difference between representation and value)
            parameters_identity: iterable of parameter identities if "parameters" is some subset of the full parameter tensor. The identity can be found with `parameter.identity` where `parameter` is a parameter object
        """
        if isinstance(parameters, dict):
            for P in parameters:
                if self[P].locked:
                    continue
                if as_representation:
                    self[P].representation = parameters[P]
                else:
                    self[P].value = parameters[P]
            return
        # ensure parameters are a tensor
        parameters = torch.as_tensor(
            parameters, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        # track the order of the parameters
        porder = self.parameter_order(parameters_identity=parameters_identity)

        # If parameters are provided by identity, they are individually updated
        if parameters_identity is not None:
            parameters_identity = list(parameters_identity)
            for P in porder:
                for pid in self[P].identities:
                    if pid in parameters_identity:
                        if as_representation:
                            self[P].set_representation(
                                parameters[parameters_identity.index(pid)], identity=pid
                            )
                        else:
                            self[P].set_value(
                                parameters[parameters_identity.index(pid)], identity=pid
                            )
            return

        # If parameters are provided as the full vector, they are added in bulk
        start = 0
        for P, V in zip(
            porder,
            self.parameter_vector_len(),
        ):
            if as_representation:
                self[P].representation = parameters[start : start + V].reshape(
                    self[P].representation.shape
                )
            else:
                self[P].value = parameters[start : start + V].reshape(
                    self[P].value.shape
                )
            start += V

    def set_uncertainty(
        self,
        uncertainty,
        as_representation=False,
        parameters_identity=None,
    ):
        if isinstance(uncertainty, dict):
            for P in uncertainty:
                if self[P].locked:
                    continue
                self[P].set_uncertainty(
                    uncertainty[P],
                    as_representation=as_representation,
                )
            return
        uncertainty = torch.as_tensor(
            uncertainty, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        # track the order of the parameters
        porder = self.parameter_order(parameters_identity=parameters_identity)

        # If uncertainty is provided by identity, they are individually updated
        if parameters_identity is not None:
            parameters_identity = list(parameters_identity)
            for P in porder:
                for pid in self[P].identities:
                    if pid in parameters_identity:
                        self[P].set_uncertainty(
                            uncertainty[parameters_identity.index(pid)],
                            as_representation=as_representation,
                            identity=pid,
                        )
            return

        # If uncertainty is provided as the full vector, they are added in bulk
        start = 0
        for P, V in zip(
            porder,
            self.parameter_vector_len(),
        ):
            self[P].set_uncertainty(
                uncertainty[start : start + V].reshape(self[P].representation.shape),
                as_representation=as_representation,
            )
            start += V

    def jacobian(
        self,
        parameters=None,
        as_representation=False,
        **kwargs,
    ):
        raise NotImplementedError("please use a subclass of AutoProf_Model")

    @property
    def window(self):
        try:
            if self._window is None:
                return self.target.window.copy()
            return self._window
        except AttributeError:
            if self.target is None:
                raise ValueError(
                    "This model has no target or window, these must be provided by the user"
                )
            return self.target.window.copy()

    def set_window(self, window):
        # If no window given, dont go any further
        if window is None:
            return

        # If the window is given in proper format, simply use as-is
        if isinstance(window, Window):
            self._window = window
        elif len(window) == 2:
            self._window = Window(
                origin=self.target.origin
                + torch.tensor(
                    (window[0][0], window[1][0]),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                * self.target.pixelscale,
                shape=torch.tensor(
                    (window[0][1] - window[0][0], window[1][1] - window[1][0]),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                * self.target.pixelscale,
            )
        elif len(window) == 4:
            self._window = Window(
                origin=torch.tensor(
                    (window[0], window[2]),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
                shape=torch.tensor(
                    (window[1] - window[0], window[3] - window[2]),
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                ),
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
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        assert tar is None or isinstance(tar, Target_Image)
        self._target = tar

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

    def parameter_vector_len(self, parameters_identity=None):
        param_vec_len = []
        for P in self.parameter_order(parameters_identity=parameters_identity):
            if parameters_identity is None:
                param_vec_len.append(int(np.prod(self[P].value.shape)))
            else:
                param_vec_len.append(
                    sum(pid in parameters_identity for pid in self[P].identities)
                )
        return param_vec_len

    def get_parameter_vector(self, as_representation=False, parameters_identity=None):
        parameters = torch.zeros(
            np.sum(self.parameter_vector_len(parameters_identity=parameters_identity)),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        porder = self.parameter_order(parameters_identity=parameters_identity)
        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid in self[P].identities:
                    if pid in parameters_identity:
                        if as_representation:
                            parameters[pindex] = self[P].get_representation(
                                identity=pid
                            )
                        else:
                            parameters[pindex] = self[P].get_value(identity=pid)
                        pindex += 1
            return parameters

        # If the full vector is requested, they are added in bulk
        vstart = 0
        for P, V in zip(
            porder,
            self.parameter_vector_len(),
        ):
            if as_representation:
                parameters[vstart : vstart + V] = self[P].representation
            else:
                parameters[vstart : vstart + V] = self[P].value
            vstart += V
        return parameters

    def get_parameter_name_vector(self, parameters_identity=None):
        parameters = []
        porder = self.parameter_order(parameters_identity=parameters_identity)
        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid, nid in zip(self[P].identities, self[P].names):
                    if pid in parameters_identity:
                        parameters.append(nid)
            return parameters

        # If the full vector is requested, they are added in bulk
        for P in porder:
            parameters += list(self[P].names)
        return parameters

    def get_parameter_identity_vector(self, parameters_identity=None):
        parameters = []
        vstart = 0
        for P, V in zip(
            self.parameter_order(),
            self.parameter_vector_len(),
        ):
            for pid in self[P].identities:
                if parameters_identity is None or pid in parameters_identity:
                    parameters.append(pid)
            vstart += V
        return parameters

    def transform(self, in_parameters, to_representation = True, parameters_identity = None):
        out_parameters = torch.zeros(
            np.sum(self.parameter_vector_len(parameters_identity = parameters_identity)),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        porder = self.parameter_order(parameters_identity = parameters_identity)
        
        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid in self[P].identities:
                    if pid in parameters_identity:
                        if to_representation:
                            out_parameters[pindex] = self[P].val_to_rep(in_parameters[pindex])
                        else:
                            out_parameters[pindex] = self[P].rep_to_val(in_parameters[pindex])
                        pindex += 1
            return out_parameters
        
        # If the full vector is requested, they are added in bulk
        vstart = 0
        for P, V in zip(
            porder,
            self.parameter_vector_len(),
        ):
            if to_representation:
                out_parameters[vstart : vstart + V] = self[P].val_to_rep(in_parameters[vstart : vstart + V])
            else:
                out_parameters[vstart : vstart + V] = self[P].rep_to_val(in_parameters[vstart : vstart + V])
            vstart += V
        return out_parameters

    def get_uncertainty_vector(self, as_representation=False):
        uncertanty = torch.zeros(
            np.sum(self.parameter_vector_len()),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        vstart = 0
        for P, V in zip(
            self.parameter_order(),
            self.parameter_vector_len(),
        ):
            if as_representation:
                uncertanty[vstart : vstart + V] = self[P].uncertainty_representation
            else:
                uncertanty[vstart : vstart + V] = self[P].uncertainty
            vstart += V
        return uncertanty

    def __str__(self):
        """String representation for the model."""
        return str(self.get_state())

    def get_state(self):
        state = {
            "name": self.name,
            "model_type": self.model_type,
        }
        return state

    def save(self, filename="AutoProf.yaml"):
        if filename.endswith(".yaml"):
            import yaml

            state = self.get_state()
            with open(filename, "w") as f:
                yaml.dump(state, f, indent=2)
        elif filename.endswith(".json"):
            import json

            state = self.get_state()
            with open(filename, "w") as f:
                json.dump(state, f, indent=2)
        elif filename.endswith(".hdf5"):
            import h5py

            state = self.get_state()
            with h5py.File(filename, "w") as F:
                dict_to_hdf5(F, state)
        else:
            if isinstance(filename, str) and "." in filename:
                raise ValueError(
                    f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5"
                )
            else:
                raise ValueError(
                    f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5"
                )

    @classmethod
    def load(cls, filename="AutoProf.yaml"):
        if isinstance(filename, dict):
            state = filename
        elif isinstance(filename, io.TextIOBase):
            import yaml

            state = yaml.load(filename, Loader=yaml.FullLoader)
        elif filename.endswith(".yaml"):
            import yaml

            with open(filename, "r") as f:
                state = yaml.load(f, Loader=yaml.FullLoader)
        elif filename.endswith(".json"):
            import json

            with open(filename, "r") as f:
                state = json.load(f)
        elif filename.endswith(".hdf5"):
            import h5py

            with h5py.File(filename, "r") as F:
                state = hdf5_to_dict(F)
        else:
            if isinstance(filename, str) and "." in filename:
                raise ValueError(
                    f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5"
                )
            else:
                raise ValueError(
                    f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5 or python dictionary."
                )
        return state

    @classmethod
    def List_Models(cls, useable=None):
        MODELS = all_subclasses(cls)
        if useable is not None:
            for model in list(MODELS):
                if model.useable is not useable:
                    MODELS.remove(model)
        return MODELS

    @classmethod
    def List_Model_Names(cls, useable=None):
        MODELS = cls.List_Models(useable=useable)
        names = []
        for model in MODELS:
            names.append(model.model_type)
        return list(sorted(names, key=lambda n: n[::-1]))

    def __eq__(self, other):
        return self is other

    @select_sample
    def __call__(
        self,
        image=None,
        parameters=None,
        as_representation=True,
        parameters_identity=None,
        window=None,
        **kwargs,
    ):
        if parameters is not None:
            self.set_parameters(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )

        return self.sample(image=image, window=window, **kwargs)
