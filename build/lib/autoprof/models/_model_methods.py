import numpy as np
from .parameter_object import Parameter
from ..utils.conversions.coordinates import coord_to_index, index_to_coord
from ..image import Model_Image, Target_Image, Window
from copy import deepcopy
import torch

@property
def integrate_window(self):
    use_center = (0.5 + torch.round(self["center"].value/self.target.pixelscale - 0.5))*self.target.pixelscale
    use_shape = (
        (self.integrate_window_size + 1 - (self.integrate_window_size % 2))*self.target.pixelscale,
        (self.integrate_window_size + 1 - (self.integrate_window_size % 2))*self.target.pixelscale,
    )
    return Window(center = use_center, shape = use_shape, dtype = self.dtype, device = self.device)
    
@property
def psf_window(self):
    use_center = torch.floor(self["center"].value/self.target.pixelscale)
    psf_offset = (self.psf_window_size - (self.psf_window_size % 2))/2
    psf_origin = (
        (use_center[0] - psf_offset)*self.target.pixelscale,
        (use_center[1] - psf_offset)*self.target.pixelscale,
    )
    psf_shape = (
        (psf_offset*2 + 1)*self.target.pixelscale,
        (psf_offset*2 + 1)*self.target.pixelscale,
    )
    return Window(origin = psf_origin, shape = psf_shape, dtype = self.dtype, device = self.device)


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
    if isinstance(user_specs, dict):
        for p in user_specs:
            # If the user supplied a parameter object subclass, simply use that as is
            if isinstance(user_specs[p], Parameter):
                parameter_specs[p] = user_specs[p]
            elif isinstance(user_specs[p], dict): # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])
            else:
                parameter_specs[p]["value"] = user_specs[p]
                
    return parameter_specs

def build_parameters(self):
    for p in self.parameter_specs:
        # skip special parameters, these must be handled by the model child
        if "|" in p:
            continue
        # skip if the parameter already exists
        if p in self.parameters:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters[p] = self.parameter_specs[p].to(dtype = self.dtype, device = self.device)
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters[p] = Parameter(p, dtype = self.dtype, device = self.device, **self.parameter_specs[p])
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")

def __str__(self):
    state = self.get_state()
    presentation = ""
    for key in state:
        presentation = presentation + f"{key}: {state[key]}\n"
    return presentation

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

def __contains__(self, key):
    try:
        self[key]
        return True
    except:
        return False
