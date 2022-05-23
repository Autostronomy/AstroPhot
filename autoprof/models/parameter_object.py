import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries, cyclic_difference

class Parameter(object):

    def __init__(self, name, **kwargs):

        self.name = name
        
        self.limits = kwargs["limits"] if "limits" in kwargs else None
        self.cyclic = kwargs["cyclic"] if "cyclic" in kwargs else False
        self.user_fixed = kwargs["fixed"] if "fixed" in kwargs else None
        self.update_fixed(False)
        if "value" in kwargs:
            self.set_value(kwargs["value"], override_fixed = True)
        else:
            self.value = None
            self.representation = None
        self.units = kwargs["units"] if "units" in kwargs else "none"
        self.uncertainty = kwargs["uncertainty"] if "uncertainty" in kwargs else None

    def update_fixed(self, fixed):
        self.fixed = fixed or bool(self.user_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if uncertainty < 0:
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
            return cyclic_difference(self.representation, other.representation, self.limits[1] - self.limits[0]).astype(np.float64)

        return (self.representation - other.representation).astype(np.float64)
