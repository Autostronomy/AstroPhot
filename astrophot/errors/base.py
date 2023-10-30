
__all__ = ("AstroPhotError", "NameNotAllowed", "SpecificationConflict")

class AstroPhotError(Exception):
    """
    Base exception for all AstroPhot processes.
    """
    ...

class NameNotAllowed(AstroPhotError):
    """
    Used for invalid names of AstroPhot objects
    """
    ...

class SpecificationConflict(AstroPhotError):
    """
    Raised when the inputs to an object are conflicting and/or ambiguous
    """
    ...
