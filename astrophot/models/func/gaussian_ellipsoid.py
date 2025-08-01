import torch


def euler_rotation_matrix(
    alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """Compute the rotation matrix from Euler angles.

    See the Z_alpha X_beta Z_gamma convention for the order of rotations here:
    https://en.wikipedia.org/wiki/Euler_angles
    """
    ca = torch.cos(alpha)
    sa = torch.sin(alpha)
    cb = torch.cos(beta)
    sb = torch.sin(beta)
    cg = torch.cos(gamma)
    sg = torch.sin(gamma)
    R = torch.stack(
        (
            torch.stack((ca * cg - cb * sa * sg, -ca * sg - cb * cg * sa, sb * sa)),
            torch.stack((cg * sa + ca * cb * sg, ca * cb * cg - sa * sg, -ca * sb)),
            torch.stack((sb * cg, sb * cg, cb)),
        ),
        dim=-1,
    )
    return R
