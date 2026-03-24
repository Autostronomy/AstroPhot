__all__ = ("AstroPhotError", "SpecificationConflict")


class AstroPhotError(Exception):
    """
    Base exception for all AstroPhot processes.
    """


class SpecificationConflict(AstroPhotError):
    """
    Raised when the inputs to an object are conflicting and/or ambiguous
    """
