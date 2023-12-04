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
from ..utils.conversions.dict_to_hdf5 import dict_to_hdf5, hdf5_to_dict
from ..utils.optimization import reduced_chi_squared
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..image import Model_Image, Window, Target_Image, Target_Image_List
from ..param import Parameter_Node
from ._shared_methods import select_target, select_sample
from .. import AP_config
from ..errors import NameNotAllowed, InvalidTarget, UnrecognizedModel, InvalidWindow

__all__ = ("AstroPhot_Model",)


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)]
    )

######################################################################
class AstroPhot_Model(object):
    """Core class for all AstroPhot models and model like objects. This
    class defines the signatures to interact with AstroPhot models
    both for users and internal functions.

    Basic usage:

    .. code-block:: python

      import astrophot as ap

      # Create a model object
      model = ap.models.AstroPhot_Model(
          name = "unique name",
          model_type = <choose a model type>,
          target = <Target_Image object>,
          window = [[a,b],[c,d]], <widnow pixel coordinates>,
          parameters = <dict of parameter specifications if desired>,
      )

      # Initialize parameters that weren't set on creation
      model.initialize()

      # Fit model to target
      result = ap.fit.lm(model, verbose=1).fit()

      # Plot the model
      fig, ax = plt.subplots()
      ap.plots.model_image(fig, ax, model)
      plt.show()

      # Sample the model
      img = model()
      pixels = img.data

    AstroPhot models are one of the main ways that one interacts with
    the code, either by setting model parameters or passing models to
    other objects, one can perform a huge variety of fitting
    tasks. The subclass `Component_Model` should be thought of as the
    basic unit when constructing a model of an image while a
    `Group_Model` is a composite structure that may represent a
    complex object, a region of an image, or even a model spanning
    many images. Constructing the `Component_Model`s is where most
    work goes, these store the actual parameters that will be
    optimized. It is important to remmeber that a `Component_Model`
    only ever applies to a single image and a single component (star,
    galaxy, or even sub-component of one of those) in that image.

    A complex representation is made by stacking many
    `Component_Model`s together, in total this may result in a very
    large number of parameters. Trying to find starting values for all
    of these parameters can be tedious and error prone, so instead all
    built-in AstroPhot models can self initialize and find reasonable
    starting parameters for most situations. Even still one may find
    that for extremely complex fits, it is more stable to first run an
    iterative fitter before global optimization to start the models in
    better initial positions.

    Args:
        name (Optional[str]): every AstroPhot model should have a unique name
        model_type (str): a model type string can determine which kind of AstroPhot model is instantiated.
        target (Optional[Target_Image]): A Target_Image object which stores information about the image which the model is trying to fit.
        filename (Optional[str]): name of a file to load AstroPhot parameters, window, and name. The model will still need to be told its target, device, and other information

    """

    model_type = "model"
    default_uncertainty = 1e-2 # During initialization, uncertainty will be assumed 1% of initial value if no uncertainty is given
    useable = False
    model_names = []

    def __new__(cls, *, filename=None, model_type=None, **kwargs):
        if filename is not None:
            state = AstroPhot_Model.load(filename)
            MODELS = AstroPhot_Model.List_Models()
            for M in MODELS:
                if M.model_type == state["model_type"]:
                    return super(AstroPhot_Model, cls).__new__(M)
            else:
                raise UnrecognizedModel(
                    f"Unknown AstroPhot model type: {state['model_type']}"
                )
        elif model_type is not None:
            MODELS = AstroPhot_Model.List_Models()  # all_subclasses(AstroPhot_Model)
            for M in MODELS:
                if M.model_type == model_type:
                    return super(AstroPhot_Model, cls).__new__(M)
            else:
                raise UnrecognizedModel(f"Unknown AstroPhot model type: {model_type}")

        return super().__new__(cls)

    def __init__(self, *, name=None, target=None, window=None, locked=False, **kwargs):
        if not hasattr(self, "_window"):
            self._window = None
        if not hasattr(self, "_target"):
            self._target = None
        self.name = name
        AP_config.ap_logger.debug("Creating model named: {self.name}")
        self.parameters = Parameter_Node(self.name)
        self.target = target
        self.window = window
        self._locked = locked
        self.mask = kwargs.get("mask", None)

    @property
    def name(self):
        """The name for this model as a string. The name should be unique
        though this is not enforced here. The name should not contain
        the `|` or `:` characters as these are reserved for internal
        use. If one tries to set the name of a model as `None` (for
        example by not providing a name for the model) then a new
        unique name will be generated. The unique name is just the
        model type for this model with an extra unique id appended to
        the end in the format of `[#]` where `#` is a number that
        increases until a unique name is found.

        """
        return self._name

    @name.setter
    def name(self, name):
        try:
            if name == self.name:
                return
        except AttributeError:
            pass
        if name is None:
            i = 0
            while True:
                proposed_name = f"{self.model_type} [{i}]"
                if proposed_name in AstroPhot_Model.model_names:
                    i += 1
                else:
                    name = proposed_name
                    break
        if ":" in name or "|" in name:
            raise NameNotAllowed("characters '|' and ':' are reserved for internal model operations please do not include these in a model name")
        self._name = name
        AstroPhot_Model.model_names.append(name)

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
        """This is called to create a blank `Model_Image` object of the
        correct format for this model. This is typically used
        internally to construct the model image before filling the
        pixel values with the model.

        """
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
        as_representation=False,
    ):
        """
        Compute the negative log likelihood of the model wrt the target image in the appropriate window. 
        """
        if parameters is not None:
            if as_representation:
                self.parameters.vector_set_representation(parameters)
            else:
                self.parameters.vector_set_values(parameters)

        model = self.sample()
        data = self.target[self.window]
        weight = data.weight
        if self.target.has_mask:
            if isinstance(data, Target_Image_List):
                mask = tuple(torch.logical_not(submask) for submask in data.mask)
                chi2 = sum(torch.sum(((mo - da).data ** 2 * wgt)[ma]) / 2.0 for mo, da, wgt, ma in zip(model, data, weight, mask))
            else:
                mask = torch.logical_not(data.mask)
                chi2 = torch.sum(((model - data).data ** 2 * weight)[mask]) / 2.0
        else:
            if isinstance(data, Target_Image_List):
                chi2 = sum(torch.sum(((mo - da).data ** 2 * wgt)) / 2.0 for mo, da, wgt in zip(model, data, weight))
            else:
                chi2 = torch.sum(((model - data).data ** 2 * weight)) / 2.0

        return chi2

    def jacobian(
        self,
        parameters=None,
        **kwargs,
    ):
        raise NotImplementedError("please use a subclass of AstroPhot_Model")

    @default_internal
    def total_flux(self, parameters=None, window=None, image=None):
        F = self(parameters = parameters, window=None, image=None)
        return torch.sum(F.data)
        
    @property
    def window(self):
        """The window defines a region on the sky in which this model will be
        optimized and typically evaluated. Two models with
        non-overlapping windows are in effect independent of each
        other. If there is another model with a window that spans both
        of them, then they are tenuously conected.

        If not provided, the model will assume a window equal to the
        target it is fitting. Note that in this case the window is not
        explicitly set to the target window, so if the model is moved
        to another target then the fitting window will also change.

        """
        if self._window is None:
            if self.target is None:
                raise ValueError(
                    "This model has no target or window, these must be provided by the user"
                )
            return self.target.window.copy()
        return self._window

    def set_window(self, window):
        if window is None:
            # If no window given, set to none
            self._window = None
        elif isinstance(window, Window):
            # If window object given, use that
            self._window = window
        elif len(window) == 2:
            # If window given in pixels, use relative to target
            self._window = self.target.window.copy().crop_to_pixel(window)
        else:
            raise InvalidWindow(f"Unrecognized window format: {str(window)}")

    @window.setter
    def window(self, window):
        self.set_window(window)

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, Target_Image)):
            raise InvalidTarget("AstroPhot_Model target must be a Target_Image instance.")
        self._target = tar

    @property
    def locked(self):
        """Set when the model should remain fixed going forward. This model
        will be bypassed when fitting parameters, however it will
        still be sampled for generating the model image.

        Warning:

          This feature is not yet fully functional and should be avoided for now. It is included here for the sake of testing.

        """
        return self._locked

    @locked.setter
    def locked(self, val):
        self._locked = val

    @property
    def parameter_order(self):
        """Returns the model parameters in the order they are kept for
        flattening, such as when evaluating the model with a tensor of
        parameter values.

        """
        return tuple(P.name for P in self.parameters)

    def __str__(self):
        """String representation for the model."""
        return self.parameters.__str__()
    
    def __repr__(self):
        """Detailed string representation for the model."""
        return yaml.dump(self.get_state(), indent=2)

    def get_state(self, *args, **kwargs):
        """Returns a dictionary of the state of the model with its name,
        type, parameters, and other important infomration. This
        dictionary is what gets saved when a model saves to disk.

        """
        state = {
            "name": self.name,
            "model_type": self.model_type,
        }
        return state

    def save(self, filename="AstroPhot.yaml"):
        """Saves a model object to disk. By default the file type should be
        yaml, this is the only file type which gets tested, though
        other file types such as json and hdf5 should work.

        """
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
    def load(cls, filename="AstroPhot.yaml"):
        """
        Loads a saved model object.
        """
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

    def __del__(self):
        try:
            i = AstroPhot_Model.model_names.index(self.name)
            AstroPhot_Model.model_names.pop(i)
        except:
            pass

    @select_sample
    def __call__(
        self,
        image=None,
        parameters=None,
        window=None,
        as_representation=False,
        **kwargs,
    ):
        
        if parameters is None:
            parameters = self.parameters
        elif isinstance(parameters, torch.Tensor):
            if as_representation:
                self.parameters.vector_set_representation(parameters)
            else:
                self.parameters.vector_set_values(parameters)
            parameters = self.parameters
        return self.sample(image=image, window=window, parameters=parameters, **kwargs)
