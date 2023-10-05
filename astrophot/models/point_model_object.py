import torch

from .model_object import Component_Model
from ..utils.decorators import default_internal


__all__ = ["Point_Model"]


class Point_Model(Component_Model):
    """Prototype point source (typically a star) model, to be subclassed
    by other point source models which define specific
    behavior. Mostly this just requires that no standard PSF
    convolution is applied to this model as that is to be handled
    internally by the model. By default, the PSF object has no
    integration mode since it will always be evaluated at the
    resolution of the PSF provided, the integration has in effect
    already been done when constructing the PSF.

    """

    model_type = f"point {Component_Model.model_type}"
    useable = False

    @property
    def psf_mode(self):
        return "none"

    @psf_mode.setter
    def psf_mode(self, val):
        pass
