import numpy as np
import torch

def boundaries(val, limits):
    """val in limits expanded to range -inf to inf

    """
    tval = val if isinstance(val, torch.Tensor) else torch.tensor(val)
    if limits[0] is None:
        return tval - 1 / (tval - limits[1])
    elif limits[1] is None:
        return tval - 1 / (tval - limits[0])
    return torch.tan((tval - limits[0]) * np.pi / (limits[1] - limits[0]) - np.pi/2)

def inv_boundaries(val, limits):
    """val in range -inf to inf compressed to within the limits

    """
    
    tval = val if isinstance(val, torch.Tensor) else torch.tensor(val)
    if limits[0] is None:
        return (tval + limits[1] - torch.sqrt(torch.pow(tval - limits[1], 2) + 4)) * 0.5
    elif limits[1] is None:
        return (tval + limits[0] + torch.sqrt(torch.pow(tval - limits[0], 2) + 4)) * 0.5
    return (torch.arctan(tval) + np.pi/2) * (limits[1] - limits[0]) / np.pi + limits[0]

def cyclic_boundaries(val, limits):
    """Applies cyclic boundary conditions to the input value.

    """
    tval = val if isinstance(val, torch.Tensor) else torch.tensor(val)
    return limits[0] + ((tval - limits[0]) % (limits[1] - limits[0]))

def cyclic_difference(val1, val2, period):
    """Applies the difference operation between two values with cyclic
    boundary conditions.

    """
    tval1 = val1 if isinstance(val1, torch.Tensor) else torch.tensor(val1)
    tval2 = val2 if isinstance(val2, torch.Tensor) else torch.tensor(val2)
    return torch.arcsin(torch.sin((tval1 - tval2) * np.pi / period)) * period / np.pi
