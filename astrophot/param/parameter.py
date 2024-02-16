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
from ..errors import InvalidParameter

__all__ = ["Parameter_Node"]
    
class Parameter_Node(Node):
    """A node representing parameters and their relative structure.

    The Parameter_Node object stores all information relevant for the
    parameters of a model. At a high level the Parameter_Node
    accomplishes two tasks. The first task is to store the actual
    parameter values, these are represented as pytorch tensors which
    can have any shape; these are leaf nodes. The second task is to
    store the relationship between parameters in a graph structure;
    these are branch nodes. The two tasks are handled by the same type
    of object since there is some overlap between them where a branch
    node acts like a leaf node in certain contexts.

    There are various quantities that a Parameter_Node tracks which
    can be provided as arguments or updated later.

    Args:
      value: The value of a node represents the tensor which will be used by models to compute their projection into the pixels of an image. These can be quite complex, see further down for more details.
      cyclic (bool): Records if the value of a node is cyclic, meaning that if it is updated outside it's limits it should be wrapped back into the limits.
      limits (Tuple[Tensor or None, Tensor or None]): Tracks if a parameter has constraints on the range of values it can take. The first element is the lower limit, the second element is the upper limit. The two elements should either be None (no limit) or tensors with the same shape as the value.
      units (str): The units of the parameter value.
      uncertainty (Tensor or None): represents the uncertainty of the parameter value. This should be None (no uncertainty) or a Tensor with the same shape as the value.
      prof (Tensor or None): This is a profile of values which has no explicit meaning, but can be used to store information which should be kept alongside the value. For example in a spline model the position of the spline points may be a ``prof`` while the flux at each node is the value to be optimized.
      shape (Tuple or None): Can be used to set the shape of the value (number of elements/dimensions). If not provided then the shape will be set by the first time a value is given. Once a shape has been set, if a value is given which cannot be coerced into that shape, then an error will be thrown.

    The ``value`` of a Parameter_Node is somewhat complicated, there
    are a number of states it can take on. The most straightforward is
    just a Tensor, if a Tensor (or just an iterable like a list or
    numpy.ndarray) is provided then the node is required to be a leaf
    node and it will store the value to be accessed later by other
    parts of AstroPhot. Another option is to set the value as another
    node (they will automatically be linked), in this case the node's
    ``value`` is just a wrapper to call for the ``value`` of the
    linked node. Finally, the value may be a function which allows for
    arbitrarily complex values to be computed from other node's
    values. The function must take as an argument the current
    Parameter_Node instance and return a Tensor. Here are some
    examples of the various ways of interacting with the ``value`` for a hypothetical parameter ``P``::

      P.value = 1. # Will create a tensor with value 1.
      P.value = P2 # calling P.value will actually call P2.value
      def compute_value(param):
        return param["P2"].value**2
      P.value = compute_value # calling P.value will call the function as: compute_value(P) which will return P2.value**2
    
    """

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
        """The ``value`` of a Parameter_Node is somewhat complicated, there
        are a number of states it can take on. The most
        straightforward is just a Tensor, if a Tensor (or just an
        iterable like a list or numpy.ndarray) is provided then the
        node is required to be a leaf node and it will store the value
        to be accessed later by other parts of AstroPhot. Another
        option is to set the value as another node (they will
        automatically be linked), in this case the node's ``value`` is
        just a wrapper to call for the ``value`` of the linked
        node. Finally, the value may be a function which allows for
        arbitrarily complex values to be computed from other node's
        values. The function must take as an argument the current
        Parameter_Node instance and return a Tensor. Here are some
        examples of the various ways of interacting with the ``value``
        for a hypothetical parameter ``P``::

          P.value = 1. # Will create a tensor with value 1.
          P.value = P2 # calling P.value will actually call P2.value
          def compute_value(param):
            return param["P2"].value**2
          P.value = compute_value # calling P.value will call the function as: compute_value(P) which will return P2.value**2

        """
        if isinstance(self._value, Parameter_Node):
            return self._value.value
        if isinstance(self._value, FunctionType):
            return self._value(self)

        return self._value

    @property
    def mask(self):
        """The mask tensor is stored internally and it cuts out some values
        from the parameter. This is used by the ``vector`` methods in
        the class to give the parameter DAG a dynamic shape.

        """
        if not self.leaf:
            return self.vector_mask()
        try:
            return self._mask
        except AttributeError:
            return torch.ones(self.shape, dtype = torch.bool, device = AP_config.ap_device)

        
    @property
    def identities(self):
        """This creates a numpy array of strings which uniquely identify
        every element in the parameter vector. For example a
        ``center`` parameter with two components [x,y] would have
        identities be ``np.array(["123456:0", "123456:1"])`` where the
        first part is the unique id for the Parameter_Node object and
        the second number indexes where in the value tensor it refers
        to.

        """
        if self.leaf:
            idstr = str(self.identity)
            return np.array(tuple(f"{idstr}:{i}" for i in range(self.size)))
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.identities for node in flat.values())
        if len(vec) > 0:
            return np.concatenate(vec)
        return np.array(())
    
    @property
    def names(self):
        """Returns a numpy array of names for all the elements of the
        ``vector`` representation where the name is determined by the
        name of the parameters. Note that this does not create a
        unique name for each element and this should only be used for
        graphical purposes on small parameter DAGs.

        """
        if self.leaf:
            S = self.size
            if S == 1:
                return np.array((self.name,))
            return np.array(tuple(f"{self.name}:{i}" for i in range(self.size)))
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.names for node in flat.values())
        if len(vec) > 0:
            return np.concatenate(vec)
        return np.array(())
    
    def vector_values(self):
        """The vector representation is for values which correspond to
        fundamental inputs to the parameter DAG. Since the DAG may
        have linked nodes, or functions which produce values derived
        from other node values, the collection of all "values" is not
        necessarily of use for some methods such as fitting
        algorithms. The vector representation is useful for optimizers
        as it gives a fundamental representation of the parameter
        DAG. The vector_values function returns a vector of the
        ``value`` for each leaf node.

        """

        if self.leaf:
            return self.value[self.mask].flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.vector_values() for node in flat.values())
        if len(vec) > 0:
            return torch.cat(vec)
        return torch.tensor((), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
    
    def vector_uncertainty(self):
        """This returns a vector (see vector_values) with the uncertainty for
        each leaf node.

        """
        if self.leaf:
            if self._uncertainty is None:
                self.uncertainty = torch.ones_like(self.value)
            return self.uncertainty[self.mask].flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.vector_uncertainty() for node in flat.values())
        if len(vec) > 0:
            return torch.cat(vec)
        return torch.tensor((), dtype = AP_config.ap_dtype, device = AP_config.ap_device)

    def vector_representation(self):
        """This returns a vector (see vector_values) with the representation
        for each leaf node. The representation is an alternative view
        of each value which is mapped into the (-inf, inf) range where
        optimization is more stable.

        """
        return self.vector_transform_val_to_rep(self.vector_values())

    def vector_mask(self):
        """This returns a vector (see vector_values) with the mask for each
        leaf node. Note however that the mask is not itself masked,
        this vector is always the full size of the unmasked parameter
        DAG.

        """
        if self.leaf:
            return self.mask.flatten()
        
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.vector_mask() for node in flat.values())
        if len(vec) > 0:
            return torch.cat(vec)
        return torch.tensor((), dtype = AP_config.ap_dtype, device = AP_config.ap_device)

    def vector_identities(self):
        """This returns a vector (see vector_values) with the identities for
        each leaf node.

        """
        if self.leaf:
            return self.identities[self.vector_mask().detach().cpu().numpy()].flatten()
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.vector_identities() for node in flat.values())
        if len(vec) > 0:
            return np.concatenate(vec)
        return np.array(())

    def vector_names(self):
        """This returns a vector (see vector_values) with the names for each
        leaf node.

        """
        if self.leaf:
            return self.names[self.vector_mask().detach().cpu().numpy()].flatten()
        flat = self.flat(include_locked = False, include_links = False)
        vec = tuple(node.vector_names() for node in flat.values())
        if len(vec) > 0:
            return np.concatenate(vec)
        return np.array(())
        
    def vector_set_values(self, values):
        """This function allows one to update the full vector of values in a
        single call by providing a tensor of the appropriate size. The
        input will be separated so that the correct elements are
        passed to the correct leaf nodes.

        """
        values = torch.as_tensor(values, dtype = AP_config.ap_dtype, device = AP_config.ap_device).flatten()
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
        """Update the uncertainty vector for this parameter DAG (see
        vector_set_values).

        """
        uncertainty = torch.as_tensor(uncertainty, dtype = AP_config.ap_dtype, device = AP_config.ap_device)        
        if self.leaf:
            if self._uncertainty is None:
                self._uncertainty = torch.ones_like(self.value)
            self._uncertainty[self.mask] = uncertainty
            return

        mask = self.vector_mask()
        flat = self.flat(include_locked = False, include_links = False)

        loc = 0
        for node in flat.values():
            node.vector_set_uncertainty(uncertainty[mask[:loc].sum().int():mask[:loc+node.size].sum().int()])
            loc += node.size

    def vector_set_mask(self, mask):
        """Update the mask vector for this parameter DAG (see
        vector_set_values). Note again that the mask vector is always
        the full size of the DAG.

        """
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
        """Update the representation vector for this parameter DAG (see
        vector_set_values).

        """        
        self.vector_set_values(self.vector_transform_rep_to_val(rep))
        
    def vector_transform_rep_to_val(self, rep):
        """Used to transform between the ``vector_values`` and
        ``vector_representation`` views of the elements in the DAG
        leafs. This transforms from representation to value.

        The transformation is done based on the limits of each
        parameter leaf. If no limits are provided then the
        representation and value are equivalent. If both are given
        then a ``tan`` and ``arctan`` are used to convert between the
        finite range and the infinite range. If the limits are
        one-sided then the transformation: ``newvalue = value - 1 /
        (value - limit)`` is used.

        """
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
        if len(vals) > 0:
            return torch.cat(vals)
        return torch.tensor((), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
    
    def vector_transform_val_to_rep(self, val):
        """Used to transform between the ``vector_values`` and
        ``vector_representation`` views of the elements in the DAG
        leafs. This transforms from value to representation.

        The transformation is done based on the limits of each
        parameter leaf. If no limits are provided then the
        representation and value are equivalent. If both are given
        then a ``tan`` and ``arctan`` are used to convert between the
        finite range and the infinite range. If the limits are
        one-sided then the transformation: ``newvalue = value - 1 /
        (value - limit)`` is used.
        
        """
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
        if len(reps) > 0:
            return torch.cat(reps)
        return torch.tensor((), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        
    def _set_val_self(self, val):
        """Handles the setting of the value for a leaf node. Ensures the
        value is a Tensor and that it has the right shape. Will also
        check the limits of the value which has different behaviour
        depending on if it is cyclic, one sided, or two sided.

        """
        val = torch.as_tensor(
            val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if self.shape is not None:
            self._value = val.reshape(self.shape)
        else:
            self._value = val
            self.shape = self._value.shape
                
        if self.cyclic:
            self._value = self.limits[0] + ((self._value - self.limits[0]) % (self.limits[1] - self.limits[0]))
            return
        if self.limits[0] is not None:
            if not torch.all(self._value > self.limits[0]):
                raise InvalidParameter(f"{self.name} has lower limit {self.limits[0].detach().cpu().tolist()}")
        if self.limits[1] is not None:
            if not torch.all(self._value < self.limits[1]):
                raise InvalidParameter(f"{self.name} has upper limit {self.limits[1].detach().cpu().tolist()}")
            
    def _soft_set_val_self(self, val):
        """The same as ``_set_val_self`` except that it doesn't raise an
        error when the values are set outside their range, instead it
        will push the values into the range defined by the limits.

        """
        val = torch.as_tensor(
            val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if self.shape is not None:
            self._value = val.reshape(self.shape)
        else:
            self._value = val
            self.shape = self._value.shape
            
        if self.cyclic:
            self._value = self.limits[0] + ((self._value - self.limits[0]) % (self.limits[1] - self.limits[0]))
            return
        if self.limits[0] is not None:
            self._value = torch.maximum(self._value, self.limits[0] + torch.ones_like(self._value) * 1e-3)
        if self.limits[1] is not None:
            self._value = torch.minimum(self._value, self.limits[1] - torch.ones_like(self._value) * 1e-3)
            
        
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
            self.vector_set_values(val)
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
        # Ensure that the uncertainty tensor has the same shape as the data
        if self.shape is not None:
            if self._uncertainty.shape != self.shape:
                self._uncertainty = self._uncertainty * torch.ones(self.shape, dtype = AP_config.ap_dtype, device = AP_config.ap_device)

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
        """Update the state of the parameter given a state variable which
        holds all information about a variable.

        """
        
        super().set_state(state)
        save_locked = self.locked
        self.locked = False
        self.units = state.get("units", None)
        self.limits = state.get("limits", (None,None))
        self.cyclic = state.get("cyclic", False)
        self.value = state.get("value", None)
        self.uncertainty = state.get("uncertainty", None)
        self.prof = state.get("prof", None)
        self.locked = save_locked

    def flat_detach(self):
        """Due to the system used to track and update values in the DAG, some
        parts of the computational graph used to determine gradients
        may linger after calling .backward on a model using the
        parameters. This function essentially resets all the leaf
        values so that the full computational graph is freed.

        """
        for P in self.flat().values():
            P.value = P.value.detach()
            if P.uncertainty is not None:
                P.uncertainty = P.uncertainty.detach()
            if P.prof is not None:
                P.prof = P.prof.detach()

    @property
    def size(self):
        if self.leaf:
            return self.value.numel()
        return self.vector_values().numel()
        
    def __len__(self):
        """The number of elements required to fully describe the DAG. This is
        the number of elements in the vector_values tensor.

        """        
        return self.size

    def print_params(self, include_locked=True, include_prof=True, include_id=True):
        if self.leaf:
            return f"{self.name}" + (f" (id-{self.identity})" if include_id else "") + f": {self.value.detach().cpu().tolist()}" + ("" if self.uncertainty is None else f" +- {self.uncertainty.detach().cpu().tolist()}") + f" [{self.units}]" + ("" if self.limits[0] is None and self.limits[1] is None else f", limits: ({None if self.limits[0] is None else self.limits[0].detach().cpu().tolist()}, {None if self.limits[1] is None else self.limits[1].detach().cpu().tolist()})") + (", cyclic" if self.cyclic else "") + (", locked" if self.locked else "") + (f", prof: {self.prof.detach().cpu().tolist()}" if include_prof and self.prof is not None else "")
        elif isinstance(self._value, Parameter_Node):
            return self.name + (f" (id-{self.identity})" if include_id else "") + " points to: " + self._value.print_params(include_locked=include_locked, include_prof=include_prof, include_id=include_id)
        return self.name + (f" (id-{self.identity}, {('function node, '+self._value.__name__) if isinstance(self._value, FunctionType) else 'branch node'})" if include_id else "") + ":\n"
        
    def __str__(self):
        reply = self.print_params(include_locked=True, include_prof=False, include_id=False)
        if self.leaf or isinstance(self._value, Parameter_Node):
            return reply
        reply += "\n".join(node.print_params(include_locked=True, include_prof=False, include_id=False) for node in self.flat(include_locked=True, include_links=False).values())
        return reply
    
    def __repr__(self, level = 0, indent = '  '):
        reply = indent*level + self.print_params(include_locked=True, include_prof=False, include_id=True)
        if self.leaf or isinstance(self._value, Parameter_Node):
            return reply
        reply += "\n".join(node.__repr__(level = level+1, indent=indent) for node in self.nodes.values())
        return reply
