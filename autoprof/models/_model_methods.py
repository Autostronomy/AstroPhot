import numpy as np
from .parameter_object import Parameter, Parameter_Array
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
from autoprof.image import Model_Image, AP_Window
from copy import deepcopy

def _set_default_parameters(self):
    self._base_window = None
    self.parameters = {}
    self.loss = None
    self.iteration = -1
    self.is_sampled = False
    self.parameter_history = []
    self.loss_history = []
    self.integrate_window = None
    self.psf_window = None

def set_target(self, target):
    self.target = target

def set_window(self, window = None, index_units = True):
    # If no window given, use the whole image
    if window is None:
        window = [
            [0, self.target.shape[0]],
            [0, self.target.shape[1]],
        ]
        index_units = False

    # If the window is given in proper format, simply use as-is
    if isinstance(window, AP_Window):
        self.window = window
    elif index_units:# If window is given as list-of-list format
        self.window = AP_Window(
            self.target.origin + np.array((window[1][0],window[0][0]))*self.target.pixelscale,
            np.array((window[1][1] - window[1][0], window[0][1] - window[0][0]))*self.target.pixelscale,
        )
    else:
        self.window = AP_Window(
            self.target.origin + np.array((window[1][0],window[0][0])),
            np.array((window[1][1] - window[1][0], window[0][1] - window[0][0])),
        )
    if self._base_window is None:
        self._base_window = deepcopy(self.window)
        
    # Create the model image for this model
    self.model_image = Model_Image(
        np.zeros(np.round(self.window.shape / self.target.pixelscale).astype(int)),
        pixelscale = self.target.pixelscale,
        window = self.window,
    )
    self.is_sampled = False
 
def scale_window(self, scale):
    self.window = self._base_window * scale
    self.window &= self.target.window
    self.set_window(self.window)

def update_integrate_window(self):
    int_origin = (
        self["center"][1].value - self.integrate_window_size*self.model_image.pixelscale/2,
        self["center"][0].value - self.integrate_window_size*self.model_image.pixelscale/2,
    )
    int_shape = (
        self.integrate_window_size*self.model_image.pixelscale,
        self.integrate_window_size*self.model_image.pixelscale,
    )
    self.integrate_window = AP_Window(origin = int_origin, shape = int_shape)
    
def update_psf_window(self):
    psf_origin = (
        self["center"][1].value - self.psf_window_size*self.model_image.pixelscale/2,
        self["center"][0].value - self.psf_window_size*self.model_image.pixelscale/2,
    )
    psf_shape = (
        self.psf_window_size*self.model_image.pixelscale,
        self.psf_window_size*self.model_image.pixelscale,
    )        
    self.psf_window = AP_Window(origin = psf_origin, shape = psf_shape)
        
def update_locked(self, locked):
    if isinstance(locked, bool):
        self.locked = bool(self.user_locked) or locked
    elif isinstance(locked, int):
        if self.user_locked is not None:
            self.locked = self.user_locked
            return
        if locked <= 0:
            self.locked = False
        else:
            self.locked = locked
    else:
        raise ValueError(f"Unrecognized lock type: {type(locked)}")

@classmethod
def build_parameter_specs(cls, user_specs = None):
    parameter_specs = {}
    for base in cls.__bases__:
        try:
            parameter_specs.update(base.build_parameter_specs())
        except AttributeError:
            pass
    parameter_specs.update(cls.parameter_specs)
    if user_specs is not None:
        for p in user_specs:
            # If the user supplied a parameter object subclass, simply use that as is
            if isinstance(user_specs[p], Parameter):
                parameter_specs[p] = user_specs[p]
            else: # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])        
    return parameter_specs

@classmethod
def build_parameter_qualities(cls):
    parameter_qualities = {}
    for base in cls.__bases__:
        try:
            parameter_qualities.update(base.build_parameter_qualities())
        except AttributeError:
            pass
    parameter_qualities.update(cls.parameter_qualities)
    return parameter_qualities

def build_parameters(self):
    for p in self.parameter_specs:
        # skip special parameters, these must be handled by the model child
        if "|" in p:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters[p] = self.parameter_specs[p]
        elif isinstance(self.parameter_specs[p], dict):
            if self.parameter_qualities[p]["form"] == "value":
                self.parameters[p] = Parameter(p, **self.parameter_specs[p])
            elif self.parameter_qualities[p]["form"] == "array":
                self.parameters[p] = Parameter_Array(p, **self.parameter_specs[p])                
            else:
                raise ValueError(f"unrecognized parameter form for {p}")
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")

def get_parameters(self, exclude_fixed = False, quality = None):
    # Return all parameters if no specifications
    if not exclude_fixed and quality is None:
        return self.parameters

    return_parameters = {}
    for p in self.parameters:
        # Skip currently fixed parameters
        if exclude_fixed and self.parameters[p].fixed:
            continue
        # Skip parameters that don't have the right qualities
        if quality is not None and self.parameter_qualities.get(quality[0], "none") != quality[1]:
            continue
        # Return parameter selected
        return_parameters[p] = self.parameters[p]
    return return_parameters

def get_loss(self):
    return self.loss

def get_parameter_history(self, limit = np.inf, exclude_fixed = True, quality = None, parameter = None, expand_arrays = True):
    if parameter is None:
        param_order = self.get_parameters(exclude_fixed = exclude_fixed, quality = quality).keys()

    parameter_history = []
    for i in range(min(limit, len(self.parameter_history))):
        sub_params = []
        if not parameter is None:
            P = self.parameter_history[i][parameter]
            if isinstance(P, Parameter_Array) and expand_arrays:
                for ip in range(len(P)):
                    sub_params.append(P[ip])
                parameter_history.append(np.array(sub_params))
            else:
                parameter_history.append(self.parameter_history[i][parameter])
            continue
        for param in param_order:
            P = self.parameter_history[i][param]
            if isinstance(P, Parameter_Array) and expand_arrays:
                for ip in range(len(P)):
                    sub_params.append(P[ip])
            else:
                sub_params.append(P)
        parameter_history.append(np.array(sub_params))
    return parameter_history                


def get_loss_history(self, limit = np.inf):

    return self.loss_history[:min(limit, len(self.loss_history))]

def get_history(self, limit = np.inf, exclude_fixed = True, quality = None):

    return self.get_loss_history(limit), self.get_parameter_history(limit = limit, exclude_fixed = exclude_fixed, quality = quality)
        
def step_iteration(self):
    if self.locked:
        if isinstance(self.locked, int):
            self.update_locked(self.locked - 1)
        if self.locked:
            return
    # Add a new set of parameters to the history that defaults to the most recent values
    if not self.loss is None:
        self.parameter_history.insert(0, deepcopy(self.parameters))
        self.loss_history.insert(0, deepcopy(self.loss))
        self.loss = None
    self.iteration += 1
    self.is_sampled = False
    self.is_convolved = False
    self.is_integrated = False

def save_model(self, fileobject):
    fileobject.write("\n" + "\n" + "*"*70 + "\n")
    fileobject.write(self.name + "\n")
    fileobject.write("*"*70 + "\n")
    for p in self.parameters:
        fileobject.write(f"{str(self.parameters[p])}\n")

def __str__(self):
    return self.name

def __getitem__(self, key):

    # Try to access the parameter by name
    if key in self.parameters:
        return self.parameters[key]

    # Check any parameter arrays for the key
    for subpar in self.parameters.values():
        if not isinstance(subpar, Parameter_Array):
            continue
        try:
            return subpar[key]
        except KeyError:
            pass
        
    raise KeyError(f"{key} not in {self.name}. {str(self)}")
