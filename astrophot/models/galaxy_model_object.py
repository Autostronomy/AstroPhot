from .model_object import ComponentModel
from .mixins import InclinedMixin
from ..utils.decorators import combine_docstrings


__all__ = ["GalaxyModel"]


@combine_docstrings
class GalaxyModel(InclinedMixin, ComponentModel):
    """Intended to represent a galaxy or extended component in an image."""

    _model_type = "galaxy"
    usable = False
