from .model_object import ComponentModel
from ..utils.decorators import combine_docstrings

__all__ = ["SkyModel"]


@combine_docstrings
class SkyModel(ComponentModel):
    """prototype class for any sky background model. This simply imposes
    that the center is a locked parameter, not involved in the
    fit. Also, a sky model object has no psf mode or integration mode
    by default since it is intended to change over much larger spatial
    scales than the psf or pixel size.

    """

    _model_type = "sky"
    usable = False

    def initialize(self):
        """Initialize the sky model, this is called after the model is
        created and before it is used. This is where we can set the
        center to be a locked parameter.
        """
        if not self.center.initialized:
            target_area = self.target[self.window]
            self.center.value = target_area.center
        super().initialize()
        self.center.to_static()

    @property
    def psf_convolve(self) -> bool:
        return False

    @psf_convolve.setter
    def psf_convolve(self, val: bool):
        pass

    @property
    def integrate_mode(self) -> str:
        return "none"

    @integrate_mode.setter
    def integrate_mode(self, val: str):
        pass
