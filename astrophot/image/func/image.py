import torch


def pixel_center_meshgrid(shape, dtype, device):
    i = torch.arange(shape[0], dtype=dtype, device=device)
    j = torch.arange(shape[1], dtype=dtype, device=device)
    return torch.meshgrid(i, j, indexing="xy")


def pixel_corner_meshgrid(shape, dtype, device):
    i = torch.arange(shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = torch.arange(shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="xy")


def pixel_simpsons_meshgrid(shape, dtype, device):
    i = 0.5 * torch.arange(2 * shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = 0.5 * torch.arange(2 * shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="xy")
