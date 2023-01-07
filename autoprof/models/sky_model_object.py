from .model_object import BaseModel

__all__ = ["Sky_Model"]

class Sky_Model(BaseModel):
    """prototype class for any sky backgorund model. This simply imposes
    that the center is a locked parameter, not involved in the
    fit. Also, a sky model object has no psf mode or integration mode
    by default since it is intended to change over much larger spatial
    scales than the psf or pixel size.

    """
    model_type = f"sky {BaseModel.model_type}"
    parameter_specs = {
        "center": {"units": "arcsec", "locked": True, "uncertainty": 0.0},
    }

    @property
    def psf_mode(self):
        return "none"
    @psf_mode.setter
    def psf_mode(self, val):
        pass
    @property
    def integrate_mode(self):
        return "none"
    @integrate_mode.setter
    def integrate_mode(self, val):
        pass
