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
    """
    Create a meshgrid for Simpson's rule integration over pixel corners.

    Parameters
    ----------
    shape : tuple
        Shape of the grid (height, width).
    dtype : torch.dtype
        Data type of the tensor.
    device : torch.device
        Device to create the tensor on.

    Returns
    -------
    tuple
        Meshgrid tensors for x and y coordinates.
    """
    i = 0.5 * torch.arange(2 * shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = 0.5 * torch.arange(2 * shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="xy")


def window_or(other_origin, self_end, other_end):

    new_origin = torch.minimum(-0.5 * torch.ones_like(other_origin), other_origin)
    new_end = torch.maximum(self_end, other_end)

    return new_origin, new_end


def window_and(other_origin, self_end, other_end):
    new_origin = torch.maximum(-0.5 * torch.ones_like(other_origin), other_origin)
    new_end = torch.minimum(self_end, other_end)

    return new_origin, new_end
