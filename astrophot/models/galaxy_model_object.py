from .model_object import ComponentModel
from .mixins import InclinedMixin


__all__ = ["GalaxyModel"]


class GalaxyModel(InclinedMixin, ComponentModel):
    """Intended to represent a galaxy or extended component in an image."""

    _model_type = "galaxy"
    usable = False
