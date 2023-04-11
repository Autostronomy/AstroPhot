import numpy as np
import torch
from typing import Optional, Union, Dict, Tuple, Any
from copy import deepcopy
from .parameter_object import Parameter
from ..utils.conversions.coordinates import coord_to_index, index_to_coord
from ..image import Model_Image, Target_Image, Window
from .. import AP_config


def integrate_window(self, image: "Image", fix_to: str = "center") -> Window:
    """
    The appropriately sized window in which to perform integration for
    this model, centered on the model center.
    Args:
    image (Model_Image): The image object to integrate.

    fix_to (str): Specifies whether to fix the window to the model center or the nearest pixel center.
                  Should be one of: center, pixel. Defaults to "center".

    Returns: Window which is The window object to integrate.

    Raises: ValueError -> If the value of fix_to is not valid.

    """
    if fix_to == "center":
        use_center = self["center"].value
    elif fix_to == "pixel":
        align = image.pixel_center_alignment()
        use_center = (
            align + torch.round(self["center"].value / image.pixelscale - align)
        ) * image.pixelscale
    else:
        raise ValueError(
            f"integrate_window fix_to should be one of: center, pixel. not {fix_to}"
        )
    window_align = torch.isclose(
        ((use_center - image.origin) / image.pixelscale) % 1,
        torch.tensor(0.5, dtype=AP_config.ap_dtype, device=AP_config.ap_device),
        atol=0.25,
    )
    request_pixels = (
        self.integrate_window_size * self.target.pixelscale / image.pixelscale
    ).to(dtype=torch.int32)
    use_shape = (
        request_pixels
        + 1
        - (request_pixels % 2)
        + 1
        - window_align.to(dtype=torch.int32)
    ) * image.pixelscale
    return Window(center=use_center, shape=use_shape)


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
    for p in self.parameter_specs:
        # skip special parameters, these must be handled by the model child
        if "|" in p:
            continue
        # skip if the parameter already exists
        if p in self.parameters:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters[p] = self.parameter_specs[p].to()
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters[p] = Parameter(p, **self.parameter_specs[p])
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
    if "|" in key and key[: key.find("|")] in self.parameters:
        return self.parameters[key[: key.find("|")]][int(key[key.find("|") + 1 :])]

    raise KeyError(f"{key} not in {self.name}. {str(self)}")


def __contains__(self, key):
    try:
        self[key]
        return True
    except:
        return False
