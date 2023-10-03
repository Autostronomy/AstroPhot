from typing import Optional
from types import FunctionType
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np

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
        if "state" in kwargs:
            return
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

    @property
    def mask(self):
        if not self.leaf:
            return self.vector_mask()
        try:
            return self._mask
        except AttributeError:
            return torch.ones(self.shape, dtype = torch.bool, device = AP_config.ap_device)

        
    def vector_values(self):
        """The vector representation is for values which correspond to
        fundamental inputs to the parameter DAG. Since the DAG may
        have linked nodes, or functions which produce values derrived
        from other node values, the collection of all "values" is not
        necessarily of use for some methods such as fitting
        algorithms.

        """

        if self.leaf:
            return self.value[self.mask].flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        return torch.cat(tuple(node.vector_values() for node in flat.values()))
    
    def vector_uncertainty(self):

        if self.leaf:
            return self.uncertainty[self.mask].flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        return torch.cat(tuple(node.vector_uncertainty() for node in flat.values()))

    def vector_representation(self):
        """The vector representation is for values which correspond to
        fundamental inputs to the parameter DAG. Since the DAG may
        have linked nodes, or functions which produce values derrived
        from other node values, the collection of all "values" is not
        necessarily of use for some methods such as fitting
        algorithms.

        """
        return self.vector_transform_val_to_rep(self.vector_values())

    def vector_mask(self):
        if self.leaf:
            return self.mask.flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        return torch.cat(tuple(node.vector_mask() for node in flat.values()))

    def vector_identities(self):
        if self.leaf:
            return self.identities[self.mask.detach().cpu().numpy()].flatten()
        flat = self.flat(include_locked = False, include_links = False)
        return np.concatenate(tuple(node.vector_identities() for node in flat.values()))

    @property
    def identities(self):
        if self.leaf:
            idstr = str(self.identity)
            return np.array(tuple(f"{idstr}:{i}" for i in range(self.size)))
        flat = self.flat(include_locked = False, include_links = False)
        return np.concatenate(tuple(node.identities() for node in flat.values()))
    
    def vector_set_values(self, values):
        values = torch.as_tensor(values, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        if self.leaf:
            self._value[self.mask] = values
            return

        mask = self.vector_mask()
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        for node in flat.values():
            node.vector_set_values(values[mask[:loc].sum().int():mask[:loc+node.size].sum().int()])
            loc += node.size
            
    def vector_set_uncertainty(self, uncertainty):
        uncertainty = torch.as_tensor(uncertainty, dtype = AP_config.ap_dtype, device = AP_config.ap_device)        
        if self.leaf:
            self._uncertainty[self.mask] = uncertainty
            return

        mask = self.vector_mask()
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        for node in flat.values():
            node.vector_set_uncertainty(uncertainty[mask[:loc].sum().int():mask[:loc+node.size].sum().int()])
            loc += node.size

    def vector_set_mask(self, mask):
        mask = torch.as_tensor(mask, dtype = torch.bool, device = AP_config.ap_device)
        if self.leaf:
            self._mask = mask.reshape(self.shape)
            return
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        for node in flat.values():
            node.vector_set_mask(mask[loc:loc+node.size])
            loc += node.size
        
    def vector_set_representation(self, rep):
        self.vector_set_values(self.vector_transform_rep_to_val(rep))
        
    def vector_transform_rep_to_val(self, rep):
        rep = torch.as_tensor(rep, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        if self.leaf:
            if self.cyclic:
                val = cyclic_boundaries(rep, (self.limits[0][self.mask], self.limits[1][self.mask]))
            elif self.limits[0] is None and self.limits[1] is None:
                val = rep
            else:
                val = inv_boundaries(
                    rep,
                    (
                        None if self.limits[0] is None else self.limits[0][self.mask],
                        None if self.limits[1] is None else self.limits[1][self.mask]
                    )
                )
            return val

        mask = self.vector_mask()
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        vals = []
        for node in flat.values():
            vals.append(node.vector_transform_rep_to_val(rep[mask[:loc].sum().int():mask[:loc+node.size].sum().int()]))
            loc += node.size
        return torch.cat(vals)
    
    def vector_transform_val_to_rep(self, val):
        val = torch.as_tensor(val, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        if self.leaf:
            if self.cyclic:
                rep = cyclic_boundaries(val, (self.limits[0][self.mask], self.limits[1][self.mask]))
            elif self.limits[0] is None and self.limits[1] is None:
                rep = val
            else:
                rep = boundaries(
                    val,
                    (
                        None if self.limits[0] is None else self.limits[0][self.mask],
                        None if self.limits[1] is None else self.limits[1][self.mask]
                    )
                )
            return rep

        mask = self.vector_mask()
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        reps = []
        for node in flat.values():
            reps.append(node.vector_transform_val_to_rep(val[mask[:loc].sum().int():mask[:loc+node.size].sum().int()]))
            loc += node.size
        return torch.cat(reps)
        
    def _set_val_subnodes(self, val):
        flat = self.flat(include_locked = False)
        loc = 0
        for node in flat.values():
            node.value = val[loc:loc + node.size]
            loc += node.size

    def _soft_set_val_self(self, val):
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
            self._value = torch.maximum(self._value, torch.ones_like(self._value) * self.limits[0] * 1.001)
        if self.limits[1] is not None:
            self._value = torch.minimum(self._value, torch.ones_like(self._value) * self.limits[1] * 0.999)
            
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
            assert torch.all(self._value > self.limits[0]), f"{self.name} has lower limit {self.limits[0].detach().cpu().tolist()}"
        if self.limits[1] is not None:
            assert torch.all(self._value < self.limits[1]), f"{self.name} has upper limit {self.limits[1].detach().cpu().tolist()}"
        
    @value.setter
    def value(self, val):
        if self.locked and not Node.global_unlock:
            return
        if val is None:
            self._value = None
            self.shape = None
            return
        if isinstance(val, str):
            self._value = val
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
        self.dump()

    @property
    def shape(self):
        try:
            if isinstance(self._value, Parameter_Node):
                return self._value.shape
            if isinstance(self._value, FunctionType):
                return self.value.shape
            if self.leaf:
                return self._shape
        except AttributeError:
            pass
        return None

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
            if isinstance(self._value, Node):
                state["value"] = "NODE:" + str(self._value.identity)
            elif isinstance(self._value, FunctionType):
                state["value"] = "FUNCTION:" + self._value.__name__
            else:
                state["value"] = self.value.detach().cpu().numpy().tolist()
        if self.shape is not None:
            state["shape"] = list(self.shape)
        if self.units is not None:
            state["units"] = self.units
        if self.uncertainty is not None:
            state["uncertainty"] = self.uncertainty.detach().cpu().numpy().tolist()
        if not (self.limits[0] is None and self.limits[1] is None):
            save_lim = []
            for i in [0, 1]:
                if self.limits[i] is None:
                    save_lim.append(None)
                else:
                    save_lim.append(self.limits[i].detach().cpu().tolist())
            state["limits"] = save_lim
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
        save_locked = self.locked
        self.locked = False
        self.units = state.get("units", None)
        self.limits = state.get("limits", (None,None))
        self.cyclic = state.get("cyclic", False)
        self.uncertainty = state.get("uncertainty", None)
        self.value = state.get("value", None)
        self.prof = state.get("prof", None)
        self.locked = save_locked

    def flat_detach(self):
        for P in self.flat().values():
            P.value = P.value.detach()
            if P.uncertainty is not None:
                P.uncertainty = P.uncertainty.detach()
            if P.prof is not None:
                P.prof = P.prof.detach()
        
    def __eq__(self, other):
        return self is other

    @property
    def size(self):
        if self.leaf:
            return self.value.numel()
        return self.vector_values().numel()
        
    def __len__(self):
        """If the parameter has multiple values, this is the length of the
        parameter.

        """        
        if self.value is None:
            return self.flat_value().numel()
        return self.value.numel()
        

    def __str__(self):
        return super().__str__() + " " + ("branch" if self.value is None else str(self.value.detach().cpu().tolist()))
    def __repr__(self):
        return super().__repr__() + "\nValue: " + ("branch" if self.value is None else str(self.value.detach().cpu().tolist()))
