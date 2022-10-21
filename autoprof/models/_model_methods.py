import numpy as np
from .parameter_object import Parameter, Pointing_Parameter
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
from autoprof.image import Model_Image, Target_Image, AP_Window
from copy import deepcopy
import torch

def _set_default_parameters(self):
    self._base_window = None
    self.parameters = {}
    self.loss = None
    self.gradient = None
    self.iteration = -1
    self.is_sampled = False
    self.center_shift = torch.zeros(2)
    
def scale_window(self, scale):
    self.window = self._base_window * scale
    self.window &= self.target.window
    self.set_window(self.window)

@property 
def target(self):
    return self._target
@target.setter
def target(self, val):
    assert isinstance(val, Target_Image)
    self._target = val
    
@property
def fit_window(self):
    return self._fit_window
def set_fit_window(self, window):
    # If no window given, use the whole image
    if window is None:
        window = [
            [0, self.target.data.shape[0]],
            [0, self.target.data.shape[1]],
        ]
        index_units = False
    
    # If the window is given in proper format, simply use as-is
    if isinstance(window, AP_Window):
        self._fit_window = window
    else:
        self._fit_window = AP_Window(
            origin = self.target.origin + np.array((window[1][0],window[0][0]))*self.target.pixelscale,
            shape = np.array((window[1][1] - window[1][0], window[0][1] - window[0][0]))*self.target.pixelscale,
            # origin = self.target.origin + np.array((window[1][0],window[0][0]))*self.target.pixelscale,
            # shape = np.array((window[0][1] - window[0][0], window[1][1] - window[1][0]))*self.target.pixelscale,
        )
    if self._base_window is None:
        self._base_window = deepcopy(self._fit_window)
        
    # Create the model image for this model
    self.model_image = Model_Image(
        pixelscale = self.target.pixelscale,
        window = self._fit_window,
    )
@fit_window.setter
def fit_window(self, window):
    self.set_fit_window(window)
    
@property
def integrate_window(self):
    use_center = self.center_shift + np.round(self["center"].value.detach().numpy()/self.model_image.pixelscale)
    int_origin = (
        (use_center[1] - (self.integrate_window_size - (self.integrate_window_size % 2))/2)*self.model_image.pixelscale,
        (use_center[0] - (self.integrate_window_size - (self.integrate_window_size % 2))/2)*self.model_image.pixelscale,
    )
    int_shape = (
        (self.integrate_window_size + 1 - (self.integrate_window_size % 2))*self.model_image.pixelscale,
        (self.integrate_window_size + 1 - (self.integrate_window_size % 2))*self.model_image.pixelscale,
    )
    return AP_Window(origin = int_origin, shape = int_shape)
    
@property
def psf_window(self):
    use_center = self.center_shift + np.round(self["center"].value.detach().numpy()/self.model_image.pixelscale)
    psf_origin = (
        (use_center[1] - (self.psf_window_size - (self.psf_window_size % 2))/2)*self.model_image.pixelscale,
        (use_center[0] - (self.psf_window_size - (self.psf_window_size % 2))/2)*self.model_image.pixelscale,
    )
    psf_shape = (
        (self.psf_window_size + 1 - (self.psf_window_size % 2))*self.model_image.pixelscale,
        (self.psf_window_size + 1 - (self.psf_window_size % 2))*self.model_image.pixelscale,
    )
    return AP_Window(origin = psf_origin, shape = psf_shape)

@property
def locked(self):
    return self._locked or self._user_locked
@locked.setter
def locked(self, val):
    assert isinstance(val, bool)
    self._locked = val

@classmethod
def build_parameter_specs(cls, user_specs = None):
    parameter_specs = {}
    for base in cls.__bases__:
        try:
            parameter_specs.update(base.build_parameter_specs())
        except AttributeError:
            pass
    parameter_specs.update(cls.parameter_specs)
    parameter_specs = deepcopy(parameter_specs)
    if user_specs is not None:
        for p in user_specs:
            # If the user supplied a parameter object subclass, simply use that as is
            if isinstance(user_specs[p], Parameter):
                parameter_specs[p] = user_specs[p]
            else: # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])        
    return parameter_specs

def build_parameters(self):
    for p in self.parameter_specs:
        # skip special parameters, these must be handled by the model child
        if "|" in p:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters[p] = self.parameter_specs[p]
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters[p] = Parameter(p, **self.parameter_specs[p])
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")

def get_parameters_representation(self, exclude_locked = True):
    return_parameters = []
    for p in self.parameters:
        # Skip currently locked parameters
        if exclude_locked and self.parameters[p].locked:
            continue
        # Return parameter selected
        return_parameters.append(self.parameters[p].representation)
    return return_parameters

def get_parameters_value(self, exclude_locked = True):
    return_parameters = {}
    for p in self.parameters:
        # Skip currently locked parameters
        if exclude_locked and self.parameters[p].locked:
            continue
        # Return parameter selected
        return_parameters[p] = self.parameters[p].value
    return return_parameters

def step_iteration(self):
    if self.locked:
        if isinstance(self.locked, int):
            self.update_locked(self.locked - 1)
        if self.locked:
            return
    if not self.loss is None:
        self.loss = None
    self.is_sampled = False
    self.iteration += 1

def save_model(self, fileobject):
    fileobject.write("\n" + "\n" + "*"*70 + "\n")
    fileobject.write(self.name + "\n")
    fileobject.write("*"*70 + "\n")
    for p in self.parameters:
        fileobject.write(f"{str(self.parameters[p])}\n")

def __str__(self):
    return self.name

def __getitem__(self, key):

    # Access an element from an array parameter
    if isinstance(key, tuple):
        return self.parameters[key[0]][key[1]]
        
    # Try to access the parameter by name
    if key in self.parameters:
        return self.parameters[key]

    # Try to get a particular element from an array parameter
    if "|" in key and key[:key.find("|")] in self.parameters:
        return self.parameters[key[:key.find("|")]][int(key[key.find("|")+1:])]
        
    raise KeyError(f"{key} not in {self.name}. {str(self)}")
