import torch

from ...utils.integration import quad_table


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


def pixel_quad_meshgrid(shape, dtype, device, order=3):
    i, j = pixel_center_meshgrid(shape, dtype, device)
    di, dj, w = quad_table(order, dtype, device)
    i = torch.repeat_interleave(i[..., None], order**2, -1) + di.flatten()
    j = torch.repeat_interleave(j[..., None], order**2, -1) + dj.flatten()
    return i, j, w.flatten()
