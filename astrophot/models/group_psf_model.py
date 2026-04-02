from .group_model_object import GroupModel
from ..image import PSFImage
from ..errors import InvalidTarget
from ..param import forward

__all__ = ["PSFGroupModel"]


class PSFGroupModel(GroupModel):
    """
    A group of PSF models. Behaves similarly to a `GroupModel`, but specifically
    designed for PSF models. Note that there is no concept of a PSFImageList, so
    they always represent a single PSF model.

    When sampling, a PSFGroupModel tells each sub-PSFModel (including nested
    sub-PSFGroupModels) to sample without normalization. This way they can fit
    with relative strengths. The final top-level PSFGroupModel will normalize
    the resulting PSF, so that the image that gets passed to the regular model
    objects for the purpose of convolution is always normalized. This means that
    the sub-PSFModels in a PSFGroupModel should have their brightness parameters
    (i.e., `I0` for the MoffatPSF) set to dynamic so they can participate in the
    fit. Though this is not strictly a requirement (say you already know the
    relative brightnesses).
    """

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
            raise InvalidTarget("GroupModel target must be a PSFImage instance.")
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = target

    @forward
    def sample(self, normalize_psf=True) -> PSFImage:
        """Sample the PSF group model on the target image."""
        image = self.target.model_image(self.window)

        for model in self.models:
            model_image = model(normalize_psf=False)
            self._ensure_vmap_compatible(image, model_image)
            image += model_image

        if normalize_psf:
            image.normalize()
        return image

    @forward
    def __call__(
        self,
        normalize_psf=True,
    ) -> PSFImage:

        return self.sample(normalize_psf=normalize_psf)
