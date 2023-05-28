from copy import copy
from time import time
import io
from typing import Optional
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

from ..utils.conversions.optimization import cyclic_difference_np
from ..utils.conversions.dict_to_hdf5 import dict_to_hdf5
from ..utils.optimization import reduced_chi_squared
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..image import Model_Image, Window, Target_Image
from .parameter_group import Parameter_Group
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
        self.parameters = Parameter_Group(self.name)
        self.requires_grad = kwargs.get("requires_grad", False)
        self.target = target
        self.window = window
        self._locked = locked
        self.mask = kwargs.get("mask", None)

    def add_equality_constraint(self, model, parameter):
        if isinstance(parameter, (tuple, list)):
            for P in parameter:
                self.add_equality_constraint(model, P)
            return
        del_param = self.parameters.get_name(parameter)
        use_param = model.parameters.get_name(parameter)
        old_groups = del_param.groups
        for group in old_groups:
            group.pop_id(del_param.identity)
            group.add_parameter(use_param)

        self.equality_constraints.append(parameter)
        model.equality_constraints.append(parameter)

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
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

    def sample(self, image=None, window=None, parameters=None, *args, **kwargs):
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
            self.parameters.set_values(
                parameters, as_representation, parameters_identity
            )

        model = self.sample()
        data = self.target[self.window]
        variance = data.variance
        if self.target.has_mask:
            mask = torch.logical_not(data.mask)
            chi2 = torch.sum(((model - data).data ** 2 / variance)[mask]) / 2.0
        else:
            chi2 = torch.sum(((model - data).data ** 2 / variance)) / 2.0

        return chi2

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

    @property
    def parameter_order(self):
        return tuple(P.name for P in self.parameters)

    def __str__(self):
        """String representation for the model."""
        return yaml.dump(self.get_state(), indent=2)

    def get_state(self):
        state = {
            "name": self.name,
            "model_type": self.model_type,
        }
        return state

    def save(self, filename="AutoProf.yaml"):
        if filename.endswith(".yaml"):
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
            state = yaml.load(filename, Loader=yaml.FullLoader)
        elif filename.endswith(".yaml"):
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

    def __getitem__(self, key):
        return self.parameters[key]

    def __contains__(self, key):
        return self.parameters.__contains__(key)

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
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, torch.Tensor):
            self.parameters.set_values(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )
            parameters = self.parameters

        return self.sample(image=image, window=window, parameters=parameters, **kwargs)
