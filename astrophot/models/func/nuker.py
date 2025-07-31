import torch


def nuker(
    R: torch.Tensor,
    Rb: torch.Tensor,
    Ib: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """Nuker 1d profile function

    **Args:**
    -  `R`: Radii tensor at which to evaluate the nuker function
    -  `Ib`: brightness at the scale length, represented as the log of the brightness divided by pixel scale squared.
    -  `Rb`: scale length radius
    -  `alpha`: sharpness of transition between power law slopes
    -  `beta`: outer power law slope
    -  `gamma`: inner power law slope

    """
    return (
        Ib
        * (2 ** ((beta - gamma) / alpha))
        * ((R / Rb) ** (-gamma))
        * ((1 + (R / Rb) ** alpha) ** ((gamma - beta) / alpha))
    )
