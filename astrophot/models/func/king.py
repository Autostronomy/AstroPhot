from ...backend_obj import backend, ArrayLike


def king(R: ArrayLike, Rc: ArrayLike, Rt: ArrayLike, alpha: ArrayLike, I0: ArrayLike) -> ArrayLike:
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
    return backend.where(
        R < Rt,
        I0 * ((backend.clamp(gamma, 0, 1) - beta) / (1 - beta)) ** alpha,
        backend.zeros_like(R),
    )
