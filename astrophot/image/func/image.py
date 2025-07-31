import torch

from ...utils.integration import quad_table


def pixel_center_meshgrid(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(shape[0], dtype=dtype, device=device)
    j = torch.arange(shape[1], dtype=dtype, device=device)
    return torch.meshgrid(i, j, indexing="ij")


def cmos_pixel_center_meshgrid(
    shape: tuple[int, int], loc: tuple[float, float], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(shape[0], dtype=dtype, device=device) + loc[0]
    j = torch.arange(shape[1], dtype=dtype, device=device) + loc[1]
    return torch.meshgrid(i, j, indexing="ij")


def pixel_corner_meshgrid(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = torch.arange(shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="ij")


def pixel_simpsons_meshgrid(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    i = 0.5 * torch.arange(2 * shape[0] + 1, dtype=dtype, device=device) - 0.5
    j = 0.5 * torch.arange(2 * shape[1] + 1, dtype=dtype, device=device) - 0.5
    return torch.meshgrid(i, j, indexing="ij")


def pixel_quad_meshgrid(
    shape: tuple[int, int], dtype: torch.dtype, device: torch.device, order=3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    i, j = pixel_center_meshgrid(shape, dtype, device)
    di, dj, w = quad_table(order, dtype, device)
    i = torch.repeat_interleave(i[..., None], order**2, -1) + di.flatten()
    j = torch.repeat_interleave(j[..., None], order**2, -1) + dj.flatten()
    return i, j, w.flatten()


def rotate(
    theta: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a rotation matrix to the X,Y coordinates
    """
    s = theta.sin()
    c = theta.cos()
    return c * x - s * y, s * x + c * y
