from .base import AstroPhotError

__all__ = ("OptimizeStopFail", "OptimizeStopSuccess")


class OptimizeStopFail(AstroPhotError):
    """
    Raised at any point to stop optimization process due to failure.
    """

    pass


class OptimizeStopSuccess(AstroPhotError):
    """
    Raised at any point to stop optimization process due to success condition.
    """

    pass
