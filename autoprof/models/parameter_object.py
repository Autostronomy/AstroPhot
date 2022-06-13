import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries, cyclic_difference

class Parameter(object):

    def __init__(self, name, **kwargs):

        self.name = name
        
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self.user_fixed = kwargs.get("fixed", None)
        self.update_fixed(False)
        self.value = None
        self.representation = None
        if "value" in kwargs:
            self.set_value(kwargs["value"], override_fixed = True)
        self.units = kwargs.get("units", "none")
        self.uncertainty = kwargs.get("uncertainty", None)

    def update_fixed(self, fixed):
        self.fixed = fixed or bool(self.user_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if np.any(uncertainty < 0):
            raise ValueError(f"{name} Uncertainty should be a positive real value, not {uncertainty}")
        self.uncertainty = uncertainty
        
    def set_value(self, value, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.cyclic:
            self.value = cyclic_boundaries(value, self.limits)
            self.representation = self.value
            return
        self.value = value
        if self.limits is None:
            self.representation = self.value
        else:
            assert self.limits[0] is None or value > self.limits[0]
            assert self.limits[1] is None or value < self.limits[1]
            self.representation = boundaries(self.value, self.limits)
        
    def set_representation(self, representation, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.limits is None or self.cyclic:
            self.set_value(representation, override_fixed)
        else:
            self.set_value(inv_boundaries(representation, self.limits), override_fixed)

    def __str__(self):
        return f"{self.name}: {self.value} +- {self.uncertainty} [{self.units}{'' if self.fixed is False else ', fixed'}{'' if self.limits is None else (', ' + str(self.limits))}{'' if self.cyclic is False else ', cyclic'}]"

    def __sub__(self, other):

        if self.cyclic:
            return cyclic_difference(self.representation, other.representation, self.limits[1] - self.limits[0])

        return self.representation - other.representation

class Parameter_Array(Parameter):
    
    def set_value(self, value, override_fixed = False, index = None):
        if self.value is None:
            self.value = []
            for i, val in enumerate(value):
                self.value.append(Parameter(
                    name = f"{self.name}:{i}",
                    limits = self.limits,
                    cyclic = self.cyclic,
                    fixed = self.user_fixed,
                    value = val,
                    units = self.units,
                    uncertainty = self.uncertainty
                ))
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_value(value[i], override_fixed)
        else:
            self.value[index].set_value(value, override_fixed)

    def get_values(self):
        return np.array(list(V.value for V in self.value))
        
    def set_representation(self, representation, override_fixed = False, index = None):
        
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_representation(representation[i], override_fixed)
        else:
            self.value[index].set_representation(representation, override_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False, index = None):
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_uncertainty(uncertainty[i], override_fixed)
        else:
            self.value[index].set_uncertainty(uncertainty, override_fixed)
        
        
    def __sub__(self, other):
        res = np.zeros(len(self.value))
        for i in range(len(self.value)):
            if isinstance(other, Parameter_Array):
                res[i] = self.value[i] - other.value[i]
            elif isinstance(other, Parameter):
                res[i] = self.value[i] - other
            else:
                raise ValueError(f"unrecognized parameter type: {type(other)}")
            
        return res

    def __getitem__(self, S):
        return self.value[S]

    def __str__(self):
        return "\n".join([f"{self.name}:"] + list(str(val) for val in self.value))
        
    def __len__(self):
        return len(self.value)
