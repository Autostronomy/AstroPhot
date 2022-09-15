import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries, cyclic_difference
from copy import deepcopy

class Parameter(object):# fixme refactor to make value a parameter stored at _value
    """Object for storing model parameters that are to be optimized in
    the fitting procedure. A value and it's uncertainty are stored as
    well as meta information about the parameter. The meta information
    indicates if the parameter has any boundary values (upper or lower
    limits) and if the parameter has cyclic boundary conditions. The
    units for the parameter are generally also specified. The
    parameter tracks if it is "fixed" meaning that it should not
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
    
    def __init__(self, name, **kwargs):

        self.name = name
        
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self.user_fixed = kwargs.get("fixed", None)
        if self.user_fixed is None:
            self.user_fixed = kwargs.get("user_fixed", None)
        self.update_fixed(False)
        self._value = None
        self._representation = None
        self.units = kwargs.get("units", "none")
        self._uncertainty = kwargs.get("uncertainty", None)
        if "value" in kwargs:
            self.set_value(kwargs["value"], override_fixed = True)

    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, val):
        self.set_value(val)
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
        
    def update_fixed(self, fixed):
        self.fixed = fixed or bool(self.user_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if np.any(uncertainty < 0):
            raise ValueError(f"{name} Uncertainty should be a positive real value, not {uncertainty}")
        self._uncertainty = uncertainty
        
    def set_value(self, val, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.cyclic:
            self._value = cyclic_boundaries(val, self.limits)
            self._representation = self._value
            return
        self._value = val
        if self.limits is None:
            self._representation = self._value
        else:
            assert self.limits[0] is None or val > self.limits[0]
            assert self.limits[1] is None or val < self.limits[1]
            self._representation = boundaries(self._value, self.limits)
        
    def set_representation(self, rep, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.limits is None or self.cyclic:
            self.set_value(rep, override_fixed)
        else:
            self.set_value(inv_boundaries(rep, self.limits), override_fixed)
    
    def __str__(self):
        return f"{self.name}: {self.value} +- {self.uncertainty} [{self.units}{'' if self.fixed is False else ', fixed'}{'' if self.limits is None else (', ' + str(self.limits))}{'' if self.cyclic is False else ', cyclic'}]"

    def __sub__(self, other):

        if self.cyclic:
            return cyclic_difference(self.representation, other.representation, self.limits[1] - self.limits[0])

        return self.representation - other.representation

class Parameter_Array(Parameter):
    """Generalizes the behavior of the parameter object to the case of
    multiple values.
    """
    def set_value(self, value, override_fixed = False, index = None):
        if self._value is None:
            self._value = []
            for i, val in enumerate(value):
                self._value.append(Parameter(
                    name = f"{self.name}:{i}",
                    limits = self.limits,
                    cyclic = self.cyclic,
                    fixed = self.user_fixed,
                    value = val,
                    units = self.units,
                    uncertainty = self.uncertainty
                ))
        elif index is None:
            for i in range(len(self.value)):
                self.value[i].set_value(value[i], override_fixed)
        else:
            self.value[index].set_value(value, override_fixed)

    @property
    def value(self):
        return np.array(list(V.value for V in self._value))
    @property
    def representation(self):
        return np.array(list(V.representation for V in self._value))
    @property
    def uncertainty(self):
        return np.array(list(V.uncertainty for V in self._value))
        
    def set_representation(self, representation, override_fixed = False, index = None):
        
        if index is None:
            for i in range(len(self.value)):
                self._value[i].set_representation(representation[i], override_fixed)
        else:
            self._value[index].set_representation(representation, override_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False, index = None):
        if index is None:
            for i in range(len(self.value)):
                self._value[i].set_uncertainty(uncertainty[i], override_fixed)
        else:
            self._value[index].set_uncertainty(uncertainty, override_fixed)
        
        
    def __sub__(self, other):
        res = np.zeros(len(self.value))
        for i in range(len(self.value)):
            if isinstance(other, Parameter_Array):
                res[i] = self._value[i] - other._value[i]
            elif isinstance(other, Parameter):
                res[i] = self._value[i] - other
            else:
                raise ValueError(f"unrecognized parameter type: {type(other)}")
            
        return res

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
            return self._value[S]
        else:
            for v in self._value:
                if S == v.name:
                    return v
            else:
                raise KeyError(f"{S} not in {self.name}. {str(self)}")

    def __str__(self):
        return "\n".join([f"{self.name}:"] + list(str(val) for val in self._value))
        
    def __len__(self):
        return len(self.value)

class Pointing_Parameter(Parameter):
    """Parameter class which simply points to another parameter
    object. This is intended for cases where a model must take
    ownership of another model, and therefore also its parameter
    objects.
    """
    
    def __init__(self, name, parameter):
        super().__init__(name, **vars(parameter))
        self.parameter = parameter

    def sync(self):
        self.update_fixed(self.parameter.fixed)
        self.set_value(self.parameter.value)
        self.set_uncertainty(self.parameter.uncertainty)
        
    def update_fixed(self, fixed):
        super().update_fixed(fixed)
        self.parameter.update_fixed(fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False):
        super().set_uncertainty(uncertainty, override_fixed)
        self.parameter.set_uncertainty(uncertainty, override_fixed)
        
    def set_value(self, value, override_fixed = False):
        super().set_value(value, override_fixed)
        self.parameter.set_value(value, override_fixed)

    def set_representation(self, representation, override_fixed = False):
        super().set_representation(representation, override_fixed)
        self.parameter.set_representation(representation, override_fixed)

class Pointing_Parameter_Array(Pointing_Parameter, Parameter_Array):
    """Parameter Array class which simply points to another parameter
    array object. This is intended for cases where a model must take
    ownership of another model, and therefore also its parameter
    objects.
    """
    
    def set_value(self, value, override_fixed = False, index = None):
        self.parameter.set_value(value, override_fixed, index)
        if self._value is None:
            self._value = []
            for i, val in enumerate(value):
                self._value.append(Pointing_Parameter(
                    name = f"{self.name}:{i}",
                    parameter = self.parameter._value[i]
                ))
        else:
            super().set_value(value, override_fixed, index)

    def set_uncertainty(self, uncertainty, override_fixed = False, index = None):
        super().set_uncertainty(uncertainty, override_fixed, index)
        self.parameter.set_uncertainty(uncertainty, override_fixed, index)
        
    def set_representation(self, representation, override_fixed = False, index = None):
        super().set_representation(representation, override_fixed, index)
        self.parameter.set_representation(representation, override_fixed, index)
