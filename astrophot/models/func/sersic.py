def sersic_n_to_b(n):
    """Compute the `b(n)` for a sersic model. This factor ensures that
    the :math:`R_e` and :math:`I_e` parameters do in fact correspond
    to the half light values and not some other scale
    radius/intensity.

    """
    x = 1 / n
    return (
        2 * n
        - 1 / 3
        + x * (4 / 405 + x * (46 / 25515 + x * (131 / 1148175 - x * 2194697 / 30690717750)))
    )


def sersic(R, n, Re, Ie):
    """Seric 1d profile function, specifically designed for pytorch
    operations

    Parameters:
        R: Radii tensor at which to evaluate the sersic function
        n: sersic index restricted to n > 0.36
        Re: Effective radius in the same units as R
        Ie: Effective surface density
    """
    bn = sersic_n_to_b(n)
    return Ie * (-bn * ((R / Re) ** (1 / n) - 1)).exp()
