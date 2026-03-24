from .base import AstroPhotError

__all__ = ("InvalidWindow", "InvalidData", "InvalidImage")


class InvalidWindow(AstroPhotError):
    """
    Raised whenever a window is misspecified
    """


class InvalidData(AstroPhotError):
    """
    Raised when the data provided to an image is invalid or cannot be processed.
    """


class InvalidImage(AstroPhotError):
    """
    Raised when an image object cannot be used as given.
    """
