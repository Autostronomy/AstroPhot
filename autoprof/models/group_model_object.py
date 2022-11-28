from .core_model import AutoProf_Model
from autoprof.image import Target_Image, Model_Image
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["Group_Model"]

class Group_Model(AutoProf_Model):
    """Model object which represents a list of other models. For each
    general AutoProf model method, this calls all the appropriate
    models from its list and combines their output into a single
    summed model.

    """

    model_type = "groupmodel"
    
    def __init__(self, name, target = None, model_list = None, locked = False, **kwargs):
        super().__init__(name, model_list, target, **kwargs)
        self.model_list = [] if model_list is None else model_list
        self.target = target
        self._user_locked = locked
        self._locked = self._user_locked
        self._psf_mode = None
        self.loss = None
        self.equality_constraints = kwargs.get("equality_constraints", None)
        if self.equality_constraints is not None and isinstance(self.equality_constraints[0], str):
            self.equality_constraints = [self.equality_constraints]
        self.update_fit_window()
        if "filename" in kwargs:
            self.load(kwargs["filename"])
        self.update_equality_constraints()

    def add_model(self, model):
        self.model_list.append(model)
        self.update_fit_window()

    def update_equality_constraints(self):
        """Equality constraints given aa a list of tuples, where each tuple is
        formatted as (parameter, model1, model2, model3, ...) which
        indicates that "parameter" will be equal across all the listed
        models.

        """
        if self.equality_constraints is None:
            return
        for constraint in self.equality_constraints:
            for model in constraint[2:]:
                self[model].parameters[constraint[0]] = self[constraint[1]].parameters[constraint[0]]
        
    def update_fit_window(self):
        self.fit_window = None
        for model in self.model_list:
            if model.locked:
                continue
            if self.fit_window is None:
                self.fit_window = deepcopy(model.fit_window)
            else:
                self.fit_window |= model.fit_window
        if self.fit_window is None:
            self.fit_window = self.target.window
        self.model_image = Model_Image(
            pixelscale = self.target.pixelscale,
            window = self.fit_window,
            dtype = self.dtype,
            device = self.device,
        )

    @property
    def parameter_order(self):
        param_order = tuple()
        for model in self.model_list:
            param_order = param_order + tuple(f"{model.name}|{mp}" for mp in model.parameter_order)
        return param_order
    
    def initialize(self):
        for model in self.model_list:
            model.initialize()
    def initialize(self):
        for model in self.model_list:
            model.locked = True
        for mi, model in enumerate(self.model_list):
            model.locked = False
            model.initialize()

    def startup(self):
        super().startup()
        for model in self.model_list:
            model.startup()
            
    def finalize(self):
        for model in self.model_list:
            model.finalize()
        
    def sample(self, sample_image = None):
        if self.locked:
            return
        if self.is_sampled and sample_image is self.model_image:
            return
        if sample_image is None or sample_image is self.model_image:
            self.model_image.clear_image()
        self.loss = None
        
        for model in self.model_list:
            if model.locked:
                continue
            model.sample(sample_image)
            if sample_image is None:
                self.model_image += model.model_image
                
        if sample_image is self.model_image:
            self.is_sampled = True

    def step(self, parameters = None, parameters_as_representation = True):
        super().step(parameters, parameters_as_representation)
        for model in self.model_list:
            model.step()
            
    def get_parameters_representation(self, exclude_locked = True, exclude_equality_constraint = True):
        all_parameters = []
        all_keys = []
        for model in self.model_list:
            keys, reps = model.get_parameters_representation(exclude_locked)
            for k, r in zip(keys, reps):
                if exclude_locked and model[k].locked:
                    continue
                if exclude_equality_constraint and self.equality_constraints is not None:
                    skip = False
                    for constraint in self.equality_constraints:
                        if k == constraint[0] and model.name in constraint[2:]:
                            skip = True
                            break
                    if skip:
                        continue
                all_parameters.append(r)
                all_keys.append(f"{model.name}|{k}")
        return all_keys, all_parameters
    
    def get_parameters_value(self, exclude_locked = True):
        all_parameters = {}
        for model in self.model_list:
            values = model.get_parameters_value(exclude_locked)
            for p in values:
                all_parameters[f"{model.name}|{p}"] = values[p] 
        return all_parameters

    def jacobian(self, parameters):
        vstart = 0
        ivstart = 0
        full_jac = torch.zeros(tuple(self.model_image.data.shape) + (len(parameters),))
        for model in self.model_list:
            keys, reps = model.get_parameters_representation()
            vend = vstart + np.sum(self.parameter_vector_len[ivstart:ivstart + len(keys)])
            sub_jac = model.jacobian(parameters[vstart:vend])
            indices = model.model_image.window.get_indices(self.model_image)
            for ip, i in enumerate(range(vstart, vend)):
                full_jac[indices[0], indices[1], i] = sub_jac[:,:,ip]
            ivstart += len(keys)
            vstart = vend
        return full_jac
            
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.model_list[key[0]][key[1]]

        if isinstance(key, str) and "|" in key:
            model_name = key[:key.find("|")]
            for model in self.model_list:
                if model.name == model_name:
                    return model[key[key.find("|")+1:]]
        elif isinstance(key, str):
            for model in self.model_list:
                if model.name == key:
                    return model
        
        raise KeyError(f"{key} not in {self.name}. {str(self)}")

    @property
    def target(self):
        return self._target
    @target.setter
    def target(self, tar):
        if tar is None:
            tar = Target_Image(data = np.zeros((100,100)), pixelscale = 1., dtype = self.dtype, device = self.device)
        assert isinstance(tar, Target_Image)
        self._target = tar.to(dtype = self.dtype, device = self.device)
        for model in self.model_list:
            model.target = tar
    @property
    def psf_mode(self):
        return self._psf_mode
    @psf_mode.setter
    def psf_mode(self, value):
        self._psf_mode = value
        for model in self.model_list:
            model.psf_mode = value
    
    def get_state(self):
        state = super().get_state()
        if "models" not in state:
            state["models"] = {}
        for model in self.model_list:
            state["models"][model.name] = model.get_state()
        return state

    def load(self, filename = "AutoProf.yaml"):
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        for model in state["models"]:
            for own_model in self.model_list:
                if model == own_model.name:
                    own_model.load(state["models"][model])
                    break
            else:
                self.add_model(AutoProf_Model(name = model, filename = state["models"][model], target = self.target))
                
    from ._model_methods import locked
