def empirical_king(R, Rc, Rt, alpha, I0):
    """
    Empirical King profile.

    Parameters
    ----------
    R : array_like
        The radial distance from the center.
    Rc : float
        The core radius of the profile.
    Rt : float
        The truncation radius of the profile.
    alpha : float
        The power-law index of the profile.
    I0 : float
        The central intensity of the profile.

    Returns
    -------
    array_like
        The intensity at each radial distance.
    """
    beta = 1 / (1 + (Rt / Rc) ** 2) ** (1 / alpha)
    gamma = 1 / (1 + (R / Rc) ** 2) ** (1 / alpha)
    return I0 * (R < Rt) * ((gamma - beta) / (1 - beta)) ** alpha
