from .group_model_object import GroupModel
from ..image import PSFImage
from ..errors import InvalidTarget
from ..param import forward

__all__ = ["PSFGroupModel"]


class PSFGroupModel(GroupModel):
    """
    A group of PSF models. Behaves similarly to a `GroupModel`, but specifically designed for PSF models.
    """

    _model_type = "psf"
    usable = True
    normalize_psf = True

    _options = ("normalize_psf",)

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
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = target

    @forward
    def sample(self, *args, **kwargs):
        """Sample the PSF group model on the target image."""
        image = self.target.model_image(self.window)

        for model in self.models:
            model_image = model()
            self._ensure_vmap_compatible(image, model_image)
            image += model_image

        if self.normalize_psf:
            image.normalize()
        return image
