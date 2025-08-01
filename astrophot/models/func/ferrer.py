import torch


def ferrer(
    R: torch.Tensor, rout: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, I0: torch.Tensor
) -> torch.Tensor:
    """
    Modified Ferrer profile.

    **Args:**
    -  `R`: Radius tensor at which to evaluate the modified Ferrer function
    -  `rout`: Outer radius of the profile
    -  `alpha`: Power-law index
    -  `beta`: Exponent for the modified Ferrer function
    -  `I0`: Central intensity
    """
    return torch.where(
        R < rout,
        I0 * ((1 - (torch.clamp(R, 0, rout) / rout) ** (2 - beta)) ** alpha),
        torch.zeros_like(R),
    )
