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
from .base import Node

__all__ = ["Parameter_Node"]
    
class Parameter_Node(Node):

    def __init__(self, name, **kwargs):

        super().__init__(name, **kwargs)
        temp_locked = self.locked
        self.locked = False
        self._value = None
        self.prof = kwargs.get("prof", None)
        self.limits = kwargs.get("limits", [None, None])
        self.cyclic = kwargs.get("cyclic", False)
        self.shape = kwargs.get("shape", None)
        self.value = kwargs.get("value", None)
        self.units = kwargs.get("units", "none")
        self.uncertainty = kwargs.get("uncertainty", None)
        self.to()
        self.locked = temp_locked

    @property
    def value(self):
        if isinstance(self._value, Parameter_Node):
            return self._value.value
        if isinstance(self._value, FunctionType):
            return self._value(self)

        return self._value

    def _set_val_subnodes(self, val):
        flat = self.flat(include_locked = False)
        loc = 0
        for node in flat.keys():
            node.value = val[loc:loc + node.size]
            loc += node.size

    def _set_val_self(self, val):
        if self.shape is not None:
            self._value = torch.as_tensor(
                val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            ).reshape(self.shape)
        else:
            self._value = torch.as_tensor(
                val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            self.shape = self._value.shape
                
        if self.cyclic:
            self._value = self.limits[0] + ((self._value - self.limits[0]) % (self.limits[1] - self.limits[0]))
        if self.limits[0] is not None:
            assert torch.all(self._value > self.limits[0])
        if self.limits[1] is not None:
            assert torch.all(self._value < self.limits[1])
        
    @value.setter
    def value(self, val):
        if self.locked and not Node.global_unlock:
            return
        if val is None:
            self._value = None
            self.shape = None
            self.dump()
            return
        if isinstance(val, Parameter_Node):
            self._value = val
            self.shape = None
            # Link only to the pointed node
            self.dump()
            self.link(val)
            return
        if isinstance(val, FunctionType):
            self._value = val
            self.shape = None
            return
        if len(self.nodes) > 0:
            self._set_val_subnodes(val)
            self.shape = None
            return
        self._set_val_self(val)
        self.dump() # fixme is this right?

    @property
    def shape(self):
        if isinstance(self._value, Parameter_Node):
            return self._value.shape
        if isinstance(self._value, FunctionType):
            return self.value.shape
        if len(self.nodes) > 0:
            return self.flat_value(include_locked = False).shape
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        
    @property
    def prof(self):
        return self._prof

    @prof.setter
    def prof(self, prof):
        if self.locked and not Node.global_unlock:
            return
        if prof is None:
            self._prof = None
            return
        self._prof = torch.as_tensor(
            prof, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    @property
    def uncertainty(self):
        return self._uncertainty
    @uncertainty.setter
    def uncertainty(self, unc):
        if self.locked and not Node.global_unlock:
            return
        if unc is None:
            self._uncertainty = None
            return
        self._uncertainty = torch.as_tensor(
            unc, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    @property
    def limits(self):
        return self._limits
    @limits.setter
    def limits(self, limits):
        if self.locked and not Node.global_unlock:
            return
        if limits[0] is None:
            low = None
        else:
            low = torch.as_tensor(
                limits[0], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        if limits[1] is None:
            high = None
        else:
            high = torch.as_tensor(
                limits[1], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
        self._limits = (low, high)

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
        elif len(self.nodes) > 0:
            for node in self.nodes.values():
                node.to(dtype, device)
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
        if self.shape is not None:
            state["shape"] = tuple(self.shape)
        if self.units is not None:
            state["units"] = self.units
        if self.uncertainty is not None:
            state["uncertainty"] = self.uncertainty.detach().cpu().numpy().tolist()
        if self.locked:
            state["locked"] = self.locked
        if not (self.limits[0] is None and self.limits[1] is None):
            save_lim = []
            for i in [0, 1]:
                if self.limits[i] is None:
                    save_lim.append(None)
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
        super().set_state(state)
        self.units = state.get("units", None)
        self.limits = state.get("limits", None)
        self.cyclic = state.get("cyclic", False)
        self.uncertainty = state.get("uncertainty", None)
        self.value = state.get("value", None)
        self.prof = state.get("prof", None)
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
        if isinstance(key, str):
            if key == self.name:
                return self
            super().__getitem__(key)
        
        raise ValueError(f"Unrecognized getitem request: {key}")
