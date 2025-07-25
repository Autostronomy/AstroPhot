def moffat(R, n, Rd, I0):
    """Moffat 1d profile function

    Parameters:
        R: Radii tensor at which to evaluate the moffat function
        n: concentration index
        Rd: scale length in the same units as R
        I0: central surface density

    """
    return I0 / (1 + (R / Rd) ** 2) ** n
