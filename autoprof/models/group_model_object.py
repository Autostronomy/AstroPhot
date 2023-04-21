from copy import deepcopy
from typing import Optional, Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt

from .core_model import AutoProf_Model
from ..image import (
    Model_Image,
    Model_Image_List,
    Target_Image,
    Image_List,
    Window,
    Window_List,
)
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["Group_Model"]


class Group_Model(AutoProf_Model):
    """Model object which represents a list of other models. For each
    general AutoProf model method, this calls all the appropriate
    models from its list and combines their output into a single
    summed model. This class shoould be used when describing any
    system more comlex than makes sense to represent with a single
    light distribution.

    Args:
        name (str): unique name for the full group model
        target (Target_Image): the target image that this group model is trying to fit to
        model_list (Optional[Sequence[AutoProf_Model]]): list of AutoProf_Model objects which will combine for the group model
        locked (bool): if the whole group of models should be locked

    """

    model_type = f"group {AutoProf_Model.model_type}"
    useable = True

    def __init__(
        self,
        name: str,
        *args,
        model_list: Optional[Sequence[AutoProf_Model]] = None,
        **kwargs,
    ):
        super().__init__(name, *args, model_list=model_list, **kwargs)
        self._param_tuple = None
        self.model_list = []
        if model_list is not None:
            self.add_model(model_list)
        self._psf_mode = "none"
        self.update_window()
        if "filename" in kwargs:
            self.load(kwargs["filename"])

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
        for mod in self.model_list:
            if model.name == mod.name:
                raise KeyError(
                    f"{self.name} already has model with name {model.name}, every model must have a unique name."
                )

        self.model_list.append(model)
        self.update_window()

    @property
    def equality_constraints(self):
        try:
            return self._equality_constraints
        except AttributeError:
            return []

    @equality_constraints.setter
    def equality_constraints(self, val):
        pass

    def pop_model(self, model):
        """Removes the specified model from the group model list. Returns the
        model object if it is found.

        """
        if isinstance(model, (tuple, list)):
            return tuple(self.remove_model(mod) for mod in model)
        if isinstance(model, str):
            for sub_model in self.model_list:
                if sub_model.name == model:
                    model = sub_model
                    break
            else:
                raise KeyError(f"Could not find {model} in {self.name} model list")

        return self.model_list.pop(self.model_list.index(model))

    def update_window(self, include_locked: bool = False):
        """Makes a new window object which encloses all the windows of the
        sub models in this group model object.

        """
        if isinstance(
            self.target, Image_List
        ):  # Window_List if target is a Target_Image_List
            new_window = [None] * len(self.target.image_list)
            for model in self.model_list:
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
                        "Group_Model cannot construct a window for itself using {type(model.target)} object. Must be a Target_Image"
                    )
            new_window = Window_List(new_window)
        else:
            new_window = None
            for model in self.model_list:
                if model.locked and not include_locked:
                    continue
                if new_window is None:
                    new_window = model.window.copy()
                else:
                    new_window |= model.window
        self.window = new_window

    def parameter_tuples_order(self, override_locked: bool = True):
        """Constructs a list where each entry is a tuple with a unique name
        for the parameter and the parameter object itself.

        """
        params = []
        self._equality_constraints = []
        for model in self.model_list:
            if model.locked and not override_locked:
                continue
            for p in model.parameters:
                if model[p].locked and not override_locked:
                    continue
                if p in model.equality_constraints:
                    for k in range(len(params)):
                        if params[k][1] is model.parameters[p]:
                            self._equality_constraints.pop(
                                self.equality_constraints.index(params[k][0])
                            )
                            params[k] = (f"{model.name}:{params[k][0]}", params[k][1])
                            self._equality_constraints.append(params[k][0])
                            break
                    else:
                        params.append((f"{model.name}|{p}", model.parameters[p]))
                        self._equality_constraints.append(f"{model.name}|{p}")
                else:
                    params.append((f"{model.name}|{p}", model.parameters[p]))
        return params

    def parameter_order(
        self, override_locked: bool = False, parameters_identity: Optional[tuple] = None
    ):
        """Gives the unique parameter names for this model in a repeatable
        order. By default, locked parameters are excluded from the
        tuple. The order of parameters will of course not be the same
        when called with override_locked True/False.

        """
        param_tuples = self.parameter_tuples_order(override_locked=override_locked)
        param_order = []
        for P, M in param_tuples:
            if parameters_identity is not None and not any(
                pid in parameters_identity for pid in self[P].identities
            ):
                continue
            param_order.append(P)

        return tuple(param_order)

    @property
    def param_tuple(self):
        """A tuple with the name of every parameter in the group model"""
        if self._param_tuple is None:
            self._param_tuple = self.parameter_tuples_order(override_locked=True)
        return self._param_tuple

    @property
    def parameters(self):
        """A dictionary in which every unique parameter appears once. This
        includes locked parameters. For constrained parameters across
        several models, the parameter will only appear once where the
        names of the models are connected by ":" characters.

        """
        try:
            return dict(P for P in self.param_tuple)
        except AttributeError:
            return {}

    @parameters.setter
    def parameters(self, val):
        """You cannot set the parameters at the group model level, this
        function exists simply to avoid raising errors when
        intializing models.

        """
        pass

    @torch.no_grad()
    @select_target
    def initialize(self, target: Optional["Target_Image"] = None):
        """
        Initialize each model in this group. Does this by iteratively initializing a model then subtracting it from a copy of the target.

        Args:
          target (Optional["Target_Image"]): A Target_Image instance to use as the source for initializing the model parameters on this image.
        """
        self._param_tuple = None

        target_copy = target.copy()
        for model in self.model_list:
            model.initialize(target_copy)
            target_copy -= model()

    def sample(
        self,
        image: Optional["Model_Image"] = None,
        window: Optional[Window] = None,
        *args,
        **kwargs,
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

        for model in self.model_list:
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
                image += model(window=use_window)
            else:
                # Will sample the entire image
                model(image, window=use_window)

        return image

    @torch.no_grad()
    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        pass_jacobian: Optional["Jacobian_Image"] = None,
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
            self.set_parameters(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.get_parameter_identity_vector(
                    parameters_identity=parameters_identity,
                )
            )
        else:
            jac_img = pass_jacobian

        for model in self.model_list:
            if isinstance(model, Group_Model):
                model.jacobian(
                    as_representation=as_representation,
                    parameters_identity=parameters_identity,
                    pass_jacobian=jac_img,
                    window=window,
                )
            else:  # fixme, maybe make pass_jacobian be filled internally to each model
                jac_img += model.jacobian(
                    as_representation=as_representation,
                    parameters_identity=parameters_identity,
                    pass_jacobian=jac_img,
                    window=window,
                )

        return jac_img

    def __getitem__(self, key):
        try:
            return self.parameters[key]
        except KeyError:
            pass

        if isinstance(key, str) and "|" in key:
            model_name = key[: key.find("|")].split(":")
            for model in self.model_list:
                if model.name in model_name:
                    return model[key[key.find("|") + 1 :]]
        elif isinstance(key, str):
            for model in self.model_list:
                if model.name == key:
                    return model

        raise KeyError(f"{key} not in {self.name}. {str(self)}")

    def __iter__(self):
        return (mod for mod in self.model_list)

    @property
    def psf_mode(self):
        return self._psf_mode

    @psf_mode.setter
    def psf_mode(self, value):
        self._psf_mode = value
        for model in self.model_list:
            model.psf_mode = value

    def get_state(self):
        """Returns a dictionary with information about the state of the model
        and its parameters.

        """
        state = super().get_state()
        if "models" not in state:
            state["models"] = {}
        for model in self.model_list:
            state["models"][model.name] = model.get_state()
        return state

    def load(self, filename="AutoProf.yaml"):
        """Loads an AutoProf state file and updates this model with the
        loaded parameters.

        """
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        for model in state["models"]:
            for own_model in self.model_list:
                if model == own_model.name:
                    own_model.load(state["models"][model])
                    break
            else:
                self.add_model(
                    AutoProf_Model(
                        name=model, filename=state["models"][model], target=self.target
                    )
                )
        self.update_window()
