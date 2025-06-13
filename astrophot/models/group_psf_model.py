from .group_model_object import Group_Model
from ..image import PSF_Image
from ..errors import InvalidTarget

__all__ = ["PSF_Group_Model"]


class PSF_Group_Model(Group_Model):

    _model_type = "psf"
    usable = True

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, target):
        if not (target is None or isinstance(target, PSF_Image)):
            raise InvalidTarget("Group_Model target must be a PSF_Image instance.")
        self._target = target
