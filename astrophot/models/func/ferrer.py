import torch
from ...backend_obj import backend, ArrayLike


def ferrer(
    R: ArrayLike, rout: ArrayLike, alpha: ArrayLike, beta: ArrayLike, I0: ArrayLike
) -> ArrayLike:
    """
    Modified Ferrer profile.

    **Args:**
    -  `R`: Radius tensor at which to evaluate the modified Ferrer function
    -  `rout`: Outer radius of the profile
    -  `alpha`: Power-law index
    -  `beta`: Exponent for the modified Ferrer function
    -  `I0`: Central intensity
    """
    return backend.where(
        R < rout,
        I0 * ((1 - (backend.clamp(R, 0, rout) / rout) ** (2 - beta)) ** alpha),
        backend.zeros_like(R),
    )
