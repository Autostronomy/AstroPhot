def modified_ferrer(R, rout, alpha, beta, I0):
    """
    Modified Ferrer profile.

    Parameters
    ----------
    R : array_like
        Radial distance from the center.
    rout : float
        Outer radius of the profile.
    alpha : float
        Power-law index.
    beta : float
        Exponent for the modified Ferrer function.
    I0 : float
        Central intensity.

    Returns
    -------
    array_like
        The modified Ferrer profile evaluated at R.
    """
    return I0 * ((1 - (R / rout) ** (2 - beta)) ** alpha) * (R < rout)
