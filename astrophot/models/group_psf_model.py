from .group_model_object import GroupModel
from ..image import PSFImage
from ..errors import InvalidTarget

__all__ = ["PSFGroupModel"]


class PSFGroupModel(GroupModel):

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
        if not (target is None or isinstance(target, PSFImage)):
            raise InvalidTarget("Group_Model target must be a PSF_Image instance.")
        self._target = target
