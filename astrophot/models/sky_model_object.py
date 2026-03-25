from .model_object import ComponentModel
from ..utils.decorators import combine_docstrings

__all__ = ["SkyModel"]


@combine_docstrings
class SkyModel(ComponentModel):
    """Prototype class for any sky background model.

    This base class imposes that the ``center`` is a locked parameter not
    involved in the fit. Sky models also have no PSF convolution or
    integration mode by default, since sky backgrounds vary on spatial
    scales much larger than the PSF or pixel size.

    """

    _parameter_specs = {
        "center": {
            "units": "arcsec",
            "shape": (2,),
            "dynamic": False,
            "description": "center of the sky model in arcsec [x, y], locked to the image center",
        }
    }
    _model_type = "sky"
    usable = False

    def __init__(self, *args, integrate_mode="none", **kwargs):
        super().__init__(*args, integrate_mode=integrate_mode, **kwargs)

    def initialize(self):
        """Initialize the sky model, this is called after the model is
        created and before it is used. This is where we can set the
        center to be a locked parameter.
        """
        if not self.center.initialized:
            target_area = self.target[self.window]
            self.center = target_area.center
        super().initialize()

    @property
    def psf_convolve(self) -> bool:
        return False

    @psf_convolve.setter
    def psf_convolve(self, val: bool):
        pass
