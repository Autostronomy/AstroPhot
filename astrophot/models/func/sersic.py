def sersic_n_to_b(n):
    """Compute the `b(n)` for a sersic model. This factor ensures that
    the :math:`R_e` and :math:`I_e` parameters do in fact correspond
    to the half light values and not some other scale
    radius/intensity.

    """

    return (
        2 * n
        + 4 / (405 * n)
        + 46 / (25515 * n**2)
        + 131 / (1148175 * n**3)
        - 2194697 / (30690717750 * n**4)
        - 1 / 3
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
    return Ie * torch.exp(-bn * (torch.pow(R / Re, 1 / n) - 1))
