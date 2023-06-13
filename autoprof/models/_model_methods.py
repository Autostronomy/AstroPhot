import numpy as np
import torch
from typing import Optional, Union, Dict, Tuple, Any
from copy import deepcopy
from .parameter_object import Parameter
from ..image import Model_Image, Target_Image, Window
from .. import AP_config


@classmethod
def build_parameter_specs(cls, user_specs=None):
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
            elif isinstance(
                user_specs[p], dict
            ):  # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])
            else:
                parameter_specs[p]["value"] = user_specs[p]

    return parameter_specs


def build_parameters(self):
    for p in self.__class__._parameter_order:
        # skip if the parameter already exists
        if p in self.parameters:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters.add_parameter(self.parameter_specs[p].to())
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters.add_parameter(Parameter(p, **self.parameter_specs[p]))
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")
