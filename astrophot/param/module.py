import numpy as np
from math import prod
from caskade import (
    Module as CModule,
    ActiveStateError,
    ParamConfigurationError,
    FillParamsArrayError,
)
from ..backend_obj import backend


class Module(CModule):

    def build_params_array_identities(self):
        identities = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.batch_shape + param.shape))
            for i in range(numel):
                identities.append(f"{id(param)}_{i}")
        return identities

    def build_params_array_uncertainty(self):
        uncertainties = []
        for param in self.dynamic_params:
            if param.uncertainty is None:
                uncertainties.append(backend.zeros_like(param.value.flatten()))
            else:
                uncertainties.append(param.uncertainty.flatten())
        return backend.concatenate(tuple(uncertainties), dim=-1)

    def build_params_array_names(self):
        names = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.batch_shape + param.shape))
            if numel == 1:
                names.append(param.name)
            else:
                for i in range(numel):
                    names.append(f"{param.name}_{i}")
        return names

    def build_params_array_units(self):
        units = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.batch_shape + param.shape))
            for _ in range(numel):
                units.append(param.units)
        return units
