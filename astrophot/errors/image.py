from .base import AstroPhotError

__all__ = ("InvalidWindow", "InvalidData")

class InvalidWindow(AstroPhotError):
    """
    Raised whenever a window is misspecified
    """
    ...

class InvalidData(AstroPhotError):
    """
    Raised when an image object can't determine the data it is holding.
    """
    ...

class InvalidImage(AstroPhotError):
    """
    Raised when an image object cannot be used as given.
    """
    ...

class InvalidWCS(AstroPhotError):
    """
    Raised when the WCS is not appropriate as given.
    """
    ...
