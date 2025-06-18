import numpy as np
from caskade import Module as CModule


class Module(CModule):

    def build_params_array_identities(self):
        identities = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.shape))
            for i in range(numel):
                identities.append(f"{id(param)}_{i}")
        return identities
