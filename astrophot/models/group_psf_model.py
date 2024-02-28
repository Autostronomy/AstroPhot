from .group_model_object import Group_Model
from ..image import PSF_Image
from ..errors import InvalidTarget

__all__ = ["PSF_Group_Model"]


class PSF_Group_Model(Group_Model):

    model_type = f"psf {Group_Model.model_type}"
    usable = True

    @property
    def psf_mode(self):
        return "none"

    @psf_mode.setter
    def psf_mode(self, value):
        pass

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, PSF_Image)):
            raise InvalidTarget("Group_Model target must be a PSF_Image instance.")
        self._target = tar

        if hasattr(self, "models"):
            for model in self.models.values():
                model.target = tar
