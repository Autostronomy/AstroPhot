import torch


def moffat(R: torch.Tensor, n: torch.Tensor, Rd: torch.Tensor, I0: torch.Tensor) -> torch.Tensor:
    """Moffat 1d profile function

    **Args:**
    -  `R`: Radii tensor at which to evaluate the moffat function
    -  `n`: concentration index
    -  `Rd`: scale length in the same units as R
    -  `I0`: central surface density

    """
    return I0 / (1 + (R / Rd) ** 2) ** n
