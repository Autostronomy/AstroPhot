from .base import AstroPhotError

__all__ = ("InvalidTarget", "UnrecognizedModel")


class InvalidTarget(AstroPhotError):
    """
    Catches when a target object is assigned incorrectly.
    """


class UnrecognizedModel(AstroPhotError):
    """
    Raised when the user tries to invoke a model that does not exist.
    """
