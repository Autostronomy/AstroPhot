from .base import AstroPhotError

__all__ = ("OptimizeStop", )

class OptimizeStop(AstroPhotError):
    """
    Raised at any point to stop optimization process.
    """
    pass
