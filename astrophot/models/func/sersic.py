import torch


C1 = 4 / 405
C2 = 46 / 25515
C3 = 131 / 1148175
C4 = -2194697 / 30690717750


def sersic_n_to_b(n: float) -> float:
    """Compute the `b(n)` for a sersic model. This factor ensures that
    the $R_e$ and $I_e$ parameters do in fact correspond
    to the half light values and not some other scale
    radius/intensity.

    """
    x = 1 / n
    return 2 * n - 1 / 3 + x * (C1 + x * (C2 + x * (C3 + C4 * x)))


def sersic(R: torch.Tensor, n: torch.Tensor, Re: torch.Tensor, Ie: torch.Tensor) -> torch.Tensor:
    """Seric 1d profile function, specifically designed for pytorch
    operations

    **Args:**
    -  `R`: Radii tensor at which to evaluate the sersic function
    -  `n`: sersic index restricted to n > 0.36
    -  `Re`: Effective radius in the same units as R
    -  `Ie`: Effective surface density
    """
    bn = sersic_n_to_b(n)
    return Ie * (-bn * ((R / Re) ** (1 / n) - 1)).exp()
