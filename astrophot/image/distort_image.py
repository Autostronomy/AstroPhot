from ..param import forward
from . import func
from ..utils.interpolate import interp2d


class DistortImageMixin:
    """
    DistortImage is a subclass of Image that applies a distortion to the image.
    This is typically used for images that have been distorted by a telescope or camera.
    """
