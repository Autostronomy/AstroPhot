from copy import deepcopy
from typing import Optional, Sequence
from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt

from .core_model import AstroPhot_Model
from ..image import (
    Image,
    Model_Image,
    Model_Image_List,
    Target_Image,
    Image_List,
    Window,
    Window_List,
    Jacobian_Image,
)
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target
from ..param import Parameter_Node
from ..errors import InvalidTarget
from .. import AP_config

__all__ = ["Group_Model"]

class Group_Model(AstroPhot_Model):
    """Model object which represents a list of other models. For each
    general AstroPhot model method, this calls all the appropriate
    models from its list and combines their output into a single
    summed model. This class shoould be used when describing any
    system more comlex than makes sense to represent with a single
    light distribution.

    Args:
        name (str): unique name for the full group model
        target (Target_Image): the target image that this group model is trying to fit to
        models (Optional[Sequence[AstroPhot_Model]]): list of AstroPhot_Model objects which will combine for the group model
        locked (bool): if the whole group of models should be locked

    """

    model_type = f"group {AstroPhot_Model.model_type}"
    useable = True

    def __init__(
        self,
            *,
        name: Optional[str] = None,
        models: Optional[Sequence[AstroPhot_Model]] = None,
        **kwargs,
    ):
        super().__init__(name=name, models=models, **kwargs)
        self._param_tuple = None
        self.models = OrderedDict()
        if models is not None:
            self.add_model(models)
        self._psf_mode = "none"
        self.update_window()
        if "filename" in kwargs:
            self.load(kwargs["filename"], new_name=name)

    def add_model(self, model):
        """Adds a new model to the group model list. Ensures that the same
        model isn't added a second time.

        Parameters:
            model: a model object to add to the model list.

        """
        if isinstance(model, (tuple, list)):
            for mod in model:
                self.add_model(mod)
            return
        if model.name in self.models and model is not self.models[model.name]:
            raise KeyError(
                f"{self.name} already has model with name {model.name}, every model must have a unique name."
            )

        self.models[model.name] = model
        self.parameters.link(model.parameters)
        self.update_window()

    def update_window(self, include_locked: bool = False):
        """Makes a new window object which encloses all the windows of the
        sub models in this group model object.

        """
        if isinstance(
            self.target, Image_List
        ):  # Window_List if target is a Target_Image_List
            new_window = [None] * len(self.target.image_list)
            for model in self.models.values():
                if model.locked and not include_locked:
                    continue
                if isinstance(model.target, Image_List):
                    for target, window in zip(model.target, model.window):
                        index = self.target.index(target)
                        if new_window[index] is None:
                            new_window[index] = window.copy()
                        else:
                            new_window[index] |= window
                elif isinstance(model.target, Target_Image):
                    index = self.target.index(model.target)
                    if new_window[index] is None:
                        new_window[index] = model.window.copy()
                    else:
                        new_window[index] |= model.window
                else:
                    raise NotImplementedError(
                        f"Group_Model cannot construct a window for itself using {type(model.target)} object. Must be a Target_Image"
                    )
            new_window = Window_List(new_window)
        else:
            new_window = None
            for model in self.models.values():
                if model.locked and not include_locked:
                    continue
                if new_window is None:
                    new_window = model.window.copy()
                else:
                    new_window |= model.window
        self.window = new_window

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self, target: Optional[Image] = None, parameters=None, **kwargs
    ):
        """
        Initialize each model in this group. Does this by iteratively initializing a model then subtracting it from a copy of the target.

        Args:
          target (Optional["Target_Image"]): A Target_Image instance to use as the source for initializing the model parameters on this image.
        """
        self._param_tuple = None
        super().initialize(target=target, parameters=parameters)

        target_copy = target.copy()
        for model in self.models.values():
            model.initialize(
                target=target_copy, parameters=parameters[model.name]
            )
            target_copy -= model(parameters=parameters[model.name])

    def sample(
        self,
        image: Optional[Image] = None,
        window: Optional[Window] = None,
        parameters: Optional["Parameter_Node"] = None,
    ):
        """Sample the group model on an image. Produces the flux values for
        each pixel associated with the models in this group. Each
        model is called individually and the results are added
        together in one larger image.

        Args:
          image (Optional["Model_Image"]): Image to sample on, overrides the windows for each sub model, they will all be evaluated over this entire image. If left as none then each sub model will be evaluated in its window.

        """
        self._param_tuple = None
        if image is None:
            sample_window = True
            image = self.make_model_image(window=window)
        else:
            sample_window = False
        if parameters is None:
            parameters = self.parameters

        for model in self.models.values():
            if window is not None and isinstance(window, Window_List):
                indices = self.target.match_indices(model.target)
                if isinstance(indices, (tuple, list)):
                    use_window = Window_List(
                        window_list=list(window.window_list[ind] for ind in indices)
                    )
                else:
                    use_window = window.window_list[indices]
            else:
                use_window = window
            if sample_window:
                # Will sample the model fit window then add to the image
                image += model(
                    window=use_window, parameters=parameters[model.name]
                )
            else:
                # Will sample the entire image
                model(
                    image, window=use_window, parameters=parameters[model.name]
                )

        return image

    @torch.no_grad()
    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
        pass_jacobian: Optional[Jacobian_Image] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
        """Compute the jacobian for this model. Done by first constructing a
        full jacobian (Npixels * Nparameters) of zeros then call the
        jacobian method of each sub model and add it in to the total.

        Args:
          parameters (Optional[torch.Tensor]): 1D parameter vector to overwrite current values
          as_representation (bool): Indiates if the "parameters" argument is in the form of the real values, or as representations in the (-inf,inf) range. Default False
          pass_jacobian (Optional["Jacobian_Image"]): A Jacobian image pre-constructed to be passed along instead of constructing new Jacobians

        """
        if window is None:
            window = self.window
        self._param_tuple = None

        if parameters is not None:
            if as_representation:
                self.parameters.vector_set_representation(parameters)
            else:
                self.parameters.vector_set_values(parameters)

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.parameters.vector_identities()
            )
        else:
            jac_img = pass_jacobian

        for model in self.models.values():
            if isinstance(model, Group_Model):
                model.jacobian(
                    as_representation=as_representation,
                    pass_jacobian=jac_img,
                    window=window,
                )
            else:  # fixme, maybe make pass_jacobian be filled internally to each model
                jac_img += model.jacobian(
                    as_representation=as_representation,
                    pass_jacobian=jac_img,
                    window=window,
                )

        return jac_img

    def __iter__(self):
        return (mod for mod in self.models.values())

    @property
    def psf_mode(self):
        return self._psf_mode

    @psf_mode.setter
    def psf_mode(self, value):
        self._psf_mode = value
        for model in self.models.values():
            model.psf_mode = value

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, Target_Image)):
            raise InvalidTarget("Group_Model target must be a Target_Image instance.")
        self._target = tar

        if hasattr(self, "models"):
            for model in self.models.values():
                model.target = tar

    def get_state(self, save_params = True):
        """Returns a dictionary with information about the state of the model
        and its parameters.

        """
        state = super().get_state(save_params = save_params)
        if save_params:
            state["parameters"] = self.parameters.get_state()
        if "models" not in state:
            state["models"] = {}
        for model in self.models.values():
            state["models"][model.name] = model.get_state(save_params = False)
        return state

    def load(self, filename="AstroPhot.yaml", new_name = None):
        """Loads an AstroPhot state file and updates this model with the
        loaded parameters.

        """
        state = AstroPhot_Model.load(filename)
        
        if new_name is None:
            new_name = state["name"]
        self.name = new_name
        
        if isinstance(state["parameters"], Parameter_Node):
            self.parameters = state["parameters"]
        else:
            self.parameters = Parameter_Node(self.name, state = state["parameters"])
            
        for model in state["models"]:
            state["models"][model]["parameters"] = self.parameters[model]
            for own_model in self.models.values():
                if model == own_model.name:
                    own_model.load(state["models"][model])
                    break
            else:
                self.add_model(
                    AstroPhot_Model(
                        name=model, filename=state["models"][model], target=self.target
                    )
                )
        self.update_window()
