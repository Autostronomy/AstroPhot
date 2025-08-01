import numpy as np
import torch
from math import prod
from caskade import (
    Module as CModule,
    ActiveStateError,
    ParamConfigurationError,
    FillDynamicParamsArrayError,
)


class Module(CModule):

    def build_params_array_identities(self):
        identities = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.shape))
            for i in range(numel):
                identities.append(f"{id(param)}_{i}")
        return identities

    def build_params_array_uncertainty(self):
        uncertainties = []
        for param in self.dynamic_params:
            if param.uncertainty is None:
                uncertainties.append(torch.zeros_like(param.value.flatten()))
            else:
                uncertainties.append(param.uncertainty.flatten())
        return torch.cat(tuple(uncertainties), dim=-1)

    def build_params_array_names(self):
        names = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.shape))
            if numel == 1:
                names.append(param.name)
            else:
                for i in range(numel):
                    names.append(f"{param.name}_{i}")
        return names

    def build_params_array_units(self):
        units = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.shape))
            for _ in range(numel):
                units.append(param.units)
        return units

    def fill_dynamic_value_uncertainties(self, uncertainty):
        if self.active:
            raise ActiveStateError(f"Cannot fill dynamic values when Module {self.name} is active")

        dynamic_params = self.dynamic_params

        if uncertainty.shape[-1] == 0:
            return  # No parameters to fill
        # check for batch dimension
        pos = 0
        for param in dynamic_params:
            if not isinstance(param.shape, tuple):
                raise ParamConfigurationError(
                    f"Param {param.name} has no shape. dynamic parameters must have a shape to use Tensor input."
                )
            # Handle scalar parameters
            size = max(1, prod(param.shape))
            try:
                val = uncertainty[..., pos : pos + size].view(param.shape)
                param.uncertainty = val
            except (RuntimeError, IndexError, ValueError, TypeError):
                raise FillDynamicParamsArrayError(self.name, uncertainty, dynamic_params)

            pos += size
        if pos != uncertainty.shape[-1]:
            raise FillDynamicParamsArrayError(self.name, uncertainty, dynamic_params)
