from .base import AstroPhotError

__all__ = ("InvalidParameter",)

class InvalidParameter(AstroPhotError):
    """
    Catches when a parameter object is assigned incorrectly.
    """
    ...
