from .core_model import AutoProf_Model
from ..image import Model_Image, Model_Image_List, Target_Image, Image_List, Window_List
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt
from .. import AP_config
__all__ = ["Group_Model"]

class Group_Model(AutoProf_Model):
    """Model object which represents a list of other models. For each
    general AutoProf model method, this calls all the appropriate
    models from its list and combines their output into a single
    summed model. This class shoould be used when describing any
    system more comlex than makes sense to represent with a single
    light distribution.

    Parameters:
        name: unique name for the full group model
        target: the target image that this group model is trying to fit to
        model_list: list of AutoProf_Model objects which will combine for the group model
        locked: boolean for if the whole group of models should be locked

    """

    model_type = "groupmodel"
    
    def __init__(self, name, *args, model_list = None, **kwargs):
        super().__init__(name, *args, model_list = model_list, **kwargs)

        self.model_list = []
        if model_list is not None:
            self.add_model(model_list)
        self._psf_mode = "none"
        self.update_window()
        if "filename" in kwargs:
            self.load(kwargs["filename"])
            
    def add_model(self, model):
        """Adds a new model to the groupmodel list. Ensures that the same
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
                raise KeyError(f"{self.name} already has model with name {model.name}, every model must have a unique name.")
            
        self.model_list.append(model)
        self.update_window()

    def pop_model(self, model):
        """Removes the specified model from the groupmodel list. Returns the
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

    def update_window(self, include_locked = False):
        """Makes a new window object which encloses all the windows of the
        sub models in this groupmodel object.

        """
        new_window = None
        for model in self.model_list:
            if model.locked and not include_locked:
                continue
            if new_window is None:
                new_window = model.window.make_copy()
            else:
                new_window |= model.window
        self.window = new_window
                
    def parameter_tuples_order(self, override_locked = False):
        """Constructs a list where each entry is a tuple with a unique name
        for the parameter and the parameter object itself.

        """
        params = []
        for model in self.model_list:
            if model.locked and not override_locked:
                continue
            for p in model.parameters:
                if model[p].locked and not override_locked:
                    continue
                if p in model.equality_constraints:
                    for k in range(len(params)):
                        if params[k][1] is model.parameters[p]:
                            params[k] = (f"{model.name}:{params[k][0]}", params[k][1])
                            break
                    else:
                        params.append((f"{model.name}|{p}", model.parameters[p]))
                else:
                    params.append((f"{model.name}|{p}", model.parameters[p]))
        return params
        
    def parameter_order(self, override_locked = True):
        """Gives the unique parameter names for this model in a repeatable
        order. By default, locked parameters are excluded from the
        tuple. The order of parameters will of course not be the same
        when called with override_locked True/False.

        """
        param_tuples = self.parameter_tuples_order(override_locked = override_locked)
        return tuple(P[0] for P in param_tuples)

    @property
    def parameters(self):
        """A dictionary in which every unique parameter appears once. This
        includes locked parameters. For constrained parameters across
        several models, the parameter will only appear once where the
        names of the models are connected by ":" characters.

        """
        try:
            param_tuples = self.parameter_tuples_order(override_locked = True)
            return dict(P for P in param_tuples)
        except AttributeError:
            return {}
        
    @torch.no_grad()
    def initialize(self, target = None):
        self.sync_target()
        if target is None:
            target = self.target
        super().initialize(target)

        target_copy = target.copy()
        for model in self.model_list:
            model.initialize(target_copy)
            target_copy -= model.sample()
            
    def finalize(self):
        """To be called after optimization. This disables any optimization
        specific properties that may be active for the collection of
        sub models.

        """
        for model in self.model_list:
            model.finalize()
        
    def make_model_image(self):
        if isinstance(self.window, Window_List):
            return Model_Image_List(list(
                Model_Image( # fixme add Model_Image __new__ method to hide list nature
                    window = window,
                    pixelscale = target.pixelscale,
                ) for window, target in zip(self.window, self.target)
            ))
        else:
            return super().make_model_image()
        
    def sample(self, sample_image = None):

        if sample_image is None:
            sample_window = True
            sample_image = self.make_model_image()
        else:
            sample_window = False
        for model in self.model_list:
            if sample_window:
                sample_image += model.sample()
            else:
                model.sample(sample_image)

        return sample_image

    def sub_model_parameter_map(self, override_locked = False):
        base_parameters = self.parameters
        base_parameters_order = self.parameter_order(override_locked = override_locked)
        base_parameter_lens = self.parameter_vector_len(override_locked = override_locked)
        param_map = []
        param_vec_map = []
        for model in self.model_list:
            sub_param_map = []
            sub_param_vec_map = []
            for P in model.parameter_order(override_locked = override_locked):
                for index in range(len(base_parameters_order)):
                    if model[P] is base_parameters[base_parameters_order[index]]:
                        sub_param_map.append(index)
                        break
                else:
                    raise RuntimeError(f"Could not find parameter {P} for model {model.name}")
                leadup = sum(base_parameter_lens[:sub_param_map[-1]])
                for i in range(base_parameter_lens[sub_param_map[-1]]):
                    sub_param_vec_map.append(leadup+i)
            param_map.append(sub_param_map)
            param_vec_map.append(sub_param_vec_map)
        return param_map, param_vec_map
        
    def jacobian(self, parameters = None, as_representation = False, override_locked = False, flatten = False):
        if isinstance(self.window, Window_List):
            raise NotImplementedError("Jacobian doesnt work for multiband models yet, it will soon")
        if parameters is not None:
            self.set_parameters(parameters, override_locked = override_locked, as_representation = as_representation)        
        param_map, param_vec_map = self.sub_model_parameter_map(override_locked = override_locked)
        full_jac = torch.zeros(tuple(self.window.get_shape_flip(self.target.pixelscale)) + (np.sum(self.parameter_vector_len(override_locked = override_locked)),), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        for model, vec_map in zip(self.model_list, param_vec_map):
            sub_jac = model.jacobian(as_representation = as_representation, override_locked = override_locked, flatten = False)
            indices = model.window._get_indices(self.window, self.target.pixelscale)
            for imodel, igroup in enumerate(vec_map):
                full_jac[indices[0], indices[1], igroup] += sub_jac[:,:,imodel]
        if flatten:
            return full_jac.reshape(-1, np.sum(self.parameter_vector_len(override_locked = override_locked)))
        return full_jac
        
    def __getitem__(self, key):
        try:
            return self.parameters[key]
        except KeyError:
            pass
        
        if isinstance(key, str) and "|" in key:
            model_name = key[:key.find("|")].split(":")
            for model in self.model_list:
                if model.name in model_name:
                    return model[key[key.find("|")+1:]]
        elif isinstance(key, str):
            for model in self.model_list:
                if model.name == key:
                    return model
        
        raise KeyError(f"{key} not in {self.name}. {str(self)}")

    def sync_target(self):
        """Ensure that the target object held by the group model matches the
        targets of the individual models that it holds.

        """
        if self._target is None:
            self.target = self.model_list[0].target
        for model in self.model_list:
            model.target = self.target
        self.update_window()
        
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

    def load(self, filename = "AutoProf.yaml"):
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
                self.add_model(AutoProf_Model(name = model, filename = state["models"][model], target = self.target))
        self.update_window()
