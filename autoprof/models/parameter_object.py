import torch
import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries, cyclic_difference

class Parameter(object):
    """Object for storing model parameters that are to be optimized in
    the fitting procedure. A value and it's uncertainty are stored as
    well as meta information about the parameter. The meta information
    indicates if the parameter has any boundary values (upper or lower
    limits) and if the parameter has cyclic boundary conditions. The
    units for the parameter are generally also specified. The
    parameter tracks if it is "locked" meaning that it should not
    change during normal fitting conditions.

    For the purpose of optimization, the parameter object also stores
    a representation of it's value which is valid in the -inf to +inf
    range. This makes the parameter better behaved with most
    optimizers as they are free to explore the full real number
    line. Any time the value, or representation are updated, both
    numbers get updated automatically so they are always in sync.

    For cyclic parameters, special care must be taken when performing
    a difference operation. The numerical difference between two
    cyclic parameters is defined as the minimum difference modulo the
    cycle length. Thus parameter objects should be allowed to handle
    differences internally. Simply take the difference between two
    parameter objects and the result will be returned properly.
    """
    
    def __init__(self, name, value = None, **kwargs):

        self.name = name
        
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self._locked = False
        self.set_value(value, override_locked = True)
        self.locked = kwargs.get("locked", False)
        self.units = kwargs.get("units", "none")
        self._uncertainty = kwargs.get("uncertainty", None)
        
        
    @property
    def value(self):
        if self._representation is None:
            return None
        if self.cyclic:
            return cyclic_boundaries(self._representation, self.limits)
        if self.limits is None:
            return self._representation
        return inv_boundaries(self._representation, self.limits)
    @value.setter
    def value(self, val):
        self.set_value(val)
    @property
    def locked(self):
        return self._locked
    @locked.setter
    def locked(self, value):
        self._locked = value
        if self._representation is not None:
            self._representation.requires_grad = not self._locked
    @property
    def representation(self):
        return self._representation
    @representation.setter
    def representation(self, rep):
        self.set_representation(rep)
    @property
    def uncertainty(self):
        return self._uncertainty
    @uncertainty.setter
    def uncertainty(self, unc):
        self.set_uncertainty(unc)
    @property
    def grad(self):
        return self._representation.grad
        
    def set_uncertainty(self, uncertainty, override_locked = False):
        if self.locked and not override_locked:
            return
        if np.any(uncertainty < 0):
            raise ValueError(f"{name} Uncertainty should be a positive real value, not {uncertainty}")
        self._uncertainty = uncertainty

    def set_value(self, val, override_locked = False, index = None):
        if self.locked and not override_locked:
            return
        if val is None:
            self._representation = None
        elif self.cyclic:
            self.set_representation(cyclic_boundaries(val, self.limits), override_locked = override_locked, index = index)
        elif self.limits is None:
            self.set_representation(val, override_locked = override_locked, index = index)
        else:
            self.set_representation(boundaries(val, self.limits), override_locked = override_locked, index = index)
        
    def set_representation(self, rep, override_locked = False, index = None):
        if self.locked and not override_locked:
            return
        if rep is None:
            self._representation = None
        elif index is None:
            self._representation = rep if isinstance(rep, torch.Tensor) else torch.tensor(rep, dtype = torch.float32)
        else:
            self._representation[index] = rep
        if self._representation is not None:
            self._representation.requires_grad = not self.locked
            
    def __str__(self):
        return f"{self.name}: {self.value} +- {self.uncertainty} [{self.units}{'' if self.locked is False else ', locked'}{'' if self.limits is None else (', ' + str(self.limits))}{'' if self.cyclic is False else ', cyclic'}]"

    def __sub__(self, other):
        if self.cyclic:
            return cyclic_difference(self.representation, other.representation, self.limits[1] - self.limits[0])

        return self.representation - other.representation

    def __iter__(self):
        self.i = -1
        return self
    def __next__(self):
        self.i += 1
        if self.i < len(self.value):
            return self[self.i]
        else:
            raise StopIteration
    
    def __getitem__(self, S):
        if isinstance(S, int):
            return self.value[S]
        if isinstance(S, str):
            return self.value[int(S[S.rfind("|")+1:])]
        return self.value
    def __len__(self):
        return len(self.value)

class Pointing_Parameter(Parameter):
    """Parameter class which simply points to another parameter
    object. This is intended for cases where a model must take
    ownership of another model, and therefore also its parameter
    objects.
    """

    def __init__(self, name, parameter):
        self.name = name
        self.parameter = parameter

    @property
    def value(self):
        return self.parameter.value
    @value.setter
    def value(self, val):
        self.parameter.set_value(val)
    @property
    def representation(self):
        return self.parameter.representation
    @representation.setter
    def representation(self, val):
        self.parameter.set_representation(val)
    @property
    def uncertainty(self):
        return self.parameter.uncertainty
    @uncertainty.setter
    def uncertainty(self, val):
        self.parameter.set_uncertainty(val)

    @property
    def cyclic(self):
        return self.parameter.cyclic
    @property
    def limits(self):
        return self.parameter.limits
    @property
    def units(self):
        return self.parameter.units
    @property
    def _value(self):
        return self.parameter._value
    @property
    def _representation(self):
        return self.parameter._representation
    @property
    def _uncertainty(self):
        return self.parameter._uncertainty
    @property
    def locked(self):
        return self.parameter.locked

    def update_locked(self, locked):
        self.parameter.update_locked(locked)
    def set_value(self, value, override_locked = False):
        self.parameter.set_value(value, override_locked)
    def set_representation(self, representation, override_locked = False):
        self.parameter.set_representation(representation, override_locked)
    def set_uncertainty(self, uncertainty, override_locked = False):
        self.parameter.set_uncertainty(uncertainty, override_locked)
    
