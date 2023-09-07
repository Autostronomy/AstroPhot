from typing import Optional
from types import FunctionType
from copy import deepcopy
from collections import OrderedDict

import torch

from ..utils.conversions.optimization import (
    boundaries,
    inv_boundaries,
    d_boundaries_dval,
    d_inv_boundaries_dval,
    cyclic_boundaries,
)
from .. import AP_config

__all__ = ["Parameter"]

class Node(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self.nodes = OrderedDict()

    def link(self, node):
        self.nodes[node.name] = node

    def unlink(self, node):
        del self.nodes[node.name]

    def dump(self):
        for name in list(self.nodes.keys()):
            del self.nodes[name]

    def __getitem__(self, key):
        if key in self.nodes:
            return self.nodes[key]
        base, stem = key.split(":", 1)
        return self.nodes[key[:base]][stem]

    def flat(self, include_locked = True):
        flat = OrderedDict()
        for node in self.nodes:
            if len(node.nodes) == 0:
                if node.locked and not include_locked:
                    continue
                flat[node] = None
            else:
                flat.update(node.flat(include_locked))
        return flat

    def get_state(self):
        state = {
            "name": self.name,
            "identity": id(self),
        }
        if len(self.nodes) > 0:
            state["nodes"] = tuple(node.get_state() for node in self.nodes.values())
        return state
    
class Param_Node(Node):

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)        

        self._prof = None
        self.limits = kwargs.get("limits", (None, None))
        self.cyclic = kwargs.get("cyclic", False)
        self.elements = kwargs.get("elements", True)
        self.locked = False
        self.value = kwargs.get("value", None)
        self.locked = kwargs.get("locked", False)
        self.units = kwargs.get("units", "none")
        self._uncertainty = None
        self.uncertainty = kwargs.get("uncertainty", None)
        self.set_profile(kwargs.get("prof", None), override_locked=True)
        self.to()

    @property
    def value(self):

        if isinstance(self._value, torch.Tensor):
            return self._value
        if isinstance(self._value, Param_Node):
            return self._value.value
        if isinstance(self._value, FunctionType):
            return self._value(self)

        return None

    @value.setter
    def value(self, val):
        if self.locked:
            return
        if val is None:
            self._value = None
            return
        if len(self.nodes) > 0:
            flat = self.flat(include_locked = False)
            loc = 0
            for node in flat.keys():
                node.value = val[loc:loc + node.size]
                loc += node.size
            return
        try:
            if self._value is None:
                self._value = torch.as_tensor(
                    val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            else:
                self._value = torch.as_tensor(
                    val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            
            if self.cyclic:
                self._value = self.limits[0] + ((self._value - self.limits[0]) % (self.limits[1] - self.limits[0]))
            if self.limits[0] is not None:
                assert torch.all(self._value > self.limits[0])
            if self.limits[1] is not None:
                assert torch.all(self._value < self.limits[1])
            return
        except:
            pass
        
        if isinstance(val, Param_Node):
            self._value = val
            # Link only to the pointed node
            self.dump()
            self.link(val)
            return
        if isinstance(val, FunctionType):
            self._value = val
            return
        raise ValueError(f"Unrecognized value type for parameter: {type(val)}")        
            
    def flat_value(self, include_locked = False):
        flat = self.flat(include_locked)
        size = 0
        for node in flat.keys():
            size += node.size

        val = torch.zeros(size)
        loc = 0
        for node in flat.keys():
            val[loc:loc + node.size] = node.value.flatten()
            loc += node.size
        return val

    def to(self, dtype=None, device=None):
        """
        updates the datatype or device of this parameter
        """
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
            
        if isinstance(self._value, torch.Tensor):
            self._value = self._value.to(dtype=dtype, device=device)
        if isinstance(self._uncertainty, torch.Tensor):
            self._uncertainty = self._uncertainty.to(dtype=dtype, device=device)
        if isinstance(self.prof, torch.Tensor):
            self.prof = self.prof.to(dtype=dtype, device=device)
        return self

    def get_state(self):
        """Return the values representing the current state of the parameter,
        this can be used to re-load the state later from memory.

        """
        state = super().get_state()
        
        if self.value is not None:
            state["value"] = self.value.detach().cpu().numpy().tolist()
        if self.units is not None:
            state["units"] = self.units
        if self.uncertainty is not None:
            state["uncertainty"] = self.uncertainty.detach().cpu().numpy().tolist()
        if self.locked:
            state["locked"] = self.locked
        if self.limits is not None:
            save_lim = []
            for i in [0, 1]:
                if self.limits[i] is None:
                    save_lim.append(None)
                elif self.limits[i].numel() == 1:
                    save_lim.append(self.limits[i].item())
                else:
                    save_lim.append(self.limits[i].detach().cpu().tolist())
            state["limits"] = tuple(save_lim)
        if self.cyclic:
            state["cyclic"] = self.cyclic
        if self.prof is not None:
            state["prof"] = self.prof.detach().cpu().tolist()

        return state
    
    def set_state(self, state):
        """Update the state of the parameter given a state variable whcih
        holds all information about a variable.

        """
        self.name = state["name"]
        self._identity = state["identity"]
        self.units = state.get("units", None)
        self.limits = state.get("limits", None)
        self.cyclic = state.get("cyclic", False)
        self.set_uncertainty(state.get("uncertainty", None), override_locked = True)
        self.set_value(state.get("value", None), override_locked = True)
        self.set_profile(state.get("prof", None), override_locked = True)
        self.locked = state.get("locked", False)

    def __eq__(self, other):
        return self is other

    @property
    def size(self):
        if len(self.nodes) > 0:
            return self.flat_value().numel()
        return self.value.numel()
        
    def __len__(self):
        """If the parameter has multiple values, this is the length of the
        parameter.

        """
        
        if self.value is None:
            return self.flat_value().numel()
        return self.value.numel()
        
    def __getitem__(self, key):
        if key == self.name:
            return self.value
        super().__getitem__(key)
