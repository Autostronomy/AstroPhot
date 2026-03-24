from ...backend_obj import ArrayLike


def moffat(R: ArrayLike, n: ArrayLike, Rd: ArrayLike, I0: ArrayLike) -> ArrayLike:
    """Moffat 1d profile function

    **Args:**
    -  `R`: Radii tensor at which to evaluate the moffat function
    -  `n`: concentration index
    -  `Rd`: scale length in the same units as R
    -  `I0`: central surface density

    """
    return I0 / (1 + (R / Rd) ** 2) ** n
