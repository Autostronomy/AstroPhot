from .base import AstroPhotError

__all__ = ("InvalidModel", "InvalidTarget", "UnrecognizedModel", "InitializationError")


class InvalidModel(AstroPhotError):
    """
    Catches when a model object is inappropriate for this instance.
    """

    ...


class InvalidTarget(AstroPhotError):
    """
    Catches when a target object is assigned incorrectly.
    """

    ...


class UnrecognizedModel(AstroPhotError):
    """
    Raised when the user tries to invoke a model that does not exist.
    """

    ...


class InitializationError(AstroPhotError):
    """
    Raised when a model cannot initialize itself.
    """

    ...
