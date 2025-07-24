import torch

from .. import config


def chi_squared(target, model, mask=None, variance=None):
    if mask is None:
        if variance is None:
            return torch.sum((target - model) ** 2)
        else:
            return torch.sum(((target - model) ** 2) / variance)
    else:
        mask = torch.logical_not(mask)
        if variance is None:
            return torch.sum((target[mask] - model[mask]) ** 2)
        else:
            return torch.sum(((target[mask] - model[mask]) ** 2) / variance[mask])


def reduced_chi_squared(target, model, params, mask=None, variance=None):
    if mask is None:
        ndf = (
            torch.prod(torch.tensor(target.shape, dtype=config.DTYPE, device=config.DEVICE))
            - params
        )
    else:
        ndf = torch.sum(torch.logical_not(mask)) - params
    return chi_squared(target, model, mask, variance) / ndf
