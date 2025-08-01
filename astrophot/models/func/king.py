import torch


def king(
    R: torch.Tensor, Rc: torch.Tensor, Rt: torch.Tensor, alpha: torch.Tensor, I0: torch.Tensor
) -> torch.Tensor:
    """
    Empirical King profile.

    **Args:**
    -  `R`: Radial distance from the center of the profile.
    -  `Rc`: Core radius of the profile.
    -  `Rt`: Truncation radius of the profile.
    -  `alpha`: Power-law index of the profile.
    -  `I0`: Central intensity of the profile.
    """
    beta = 1 / (1 + (Rt / Rc) ** 2) ** (1 / alpha)
    gamma = 1 / (1 + (R / Rc) ** 2) ** (1 / alpha)
    return torch.where(
        R < Rt, I0 * ((torch.clamp(gamma, 0, 1) - beta) / (1 - beta)) ** alpha, torch.zeros_like(R)
    )
