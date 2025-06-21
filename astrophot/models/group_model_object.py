from typing import Optional, Sequence, Union

import torch
from caskade import forward

from .base import Model
from ..image import (
    Image,
    TargetImage,
    TargetImageList,
    ModelImage,
    ModelImageList,
    ImageList,
    Window,
    WindowList,
    JacobianImage,
)
from ..utils.decorators import ignore_numpy_warnings
from ..errors import InvalidTarget

__all__ = ["GroupModel"]


class GroupModel(Model):
    """Model object which represents a list of other models. For each
    general AstroPhot model method, this calls all the appropriate
    models from its list and combines their output into a single
    summed model. This class should be used when describing any
    system more complex than makes sense to represent with a single
    light distribution.

    Args:
        name (str): unique name for the full group model
        target (Target_Image): the target image that this group model is trying to fit to
        models (Optional[Sequence[AstroPhot_Model]]): list of AstroPhot_Model objects which will combine for the group model
        locked (bool): if the whole group of models should be locked

    """

    _model_type = "group"
    usable = True

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        models: Optional[Sequence[Model]] = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.models = models
        self.update_window()

    def update_window(self):
        """Makes a new window object which encloses all the windows of the
        sub models in this group model object.

        """
        if isinstance(self.target, ImageList):  # WindowList if target is a TargetImageList
            new_window = [None] * len(self.target.images)
            for model in self.models.values():
                if isinstance(model.target, ImageList):
                    for target, window in zip(model.target, model.window):
                        index = self.target.index(target)
                        if new_window[index] is None:
                            new_window[index] = window.copy()
                        else:
                            new_window[index] |= window
                elif isinstance(model.target, TargetImage):
                    index = self.target.index(model.target)
                    if new_window[index] is None:
                        new_window[index] = model.window.copy()
                    else:
                        new_window[index] |= model.window
                else:
                    raise NotImplementedError(
                        f"Group_Model cannot construct a window for itself using {type(model.target)} object. Must be a Target_Image"
                    )
            new_window = WindowList(new_window)
        else:
            new_window = None
            for model in self.models.values():
                if new_window is None:
                    new_window = model.window.copy()
                else:
                    new_window |= model.window
        self.window = new_window

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        """
        Initialize each model in this group. Does this by iteratively initializing a model then subtracting it from a copy of the target.

        Args:
          target (Optional["Target_Image"]): A Target_Image instance to use as the source for initializing the model parameters on this image.
        """
        super().initialize()

        for model in self.models.values():
            model.initialize()

    def fit_mask(self) -> torch.Tensor:
        """Returns a mask for the target image which is the combination of all
        the fit masks of the sub models. This mask is used when the multiple
        models in the group model do not completely overlap with each other, thus
        there are some pixels which are not covered by any model and have no
        reason to be fit.

        """
        subtarget = self.target[self.window]
        if isinstance(self.target, ImageList):
            mask = tuple(torch.ones_like(submask) for submask in subtarget.mask)
            for model in self.models.values():
                model_subtarget = model.target[model.window]
                model_fit_mask = model.fit_mask()
                if isinstance(model.target, ImageList):
                    for target, submask in zip(model_subtarget, model_fit_mask):
                        index = subtarget.index(target)
                        group_indices = subtarget.images[index].get_indices(target)
                        model_indices = target.get_indices(subtarget.images[index])
                        mask[index][group_indices] &= submask[model_indices]
                else:
                    index = subtarget.index(model_subtarget)
                    group_indices = subtarget.images[index].get_indices(model_subtarget)
                    model_indices = model_subtarget.get_indices(subtarget.images[index])
                    mask[index][group_indices] &= model_fit_mask[model_indices]
        else:
            mask = torch.ones_like(subtarget.mask)
            for model in self.models.values():
                model_subtarget = model.target[model.window]
                group_indices = subtarget.get_indices(model_subtarget)
                model_indices = model_subtarget.get_indices(subtarget)
                mask[group_indices] &= model.fit_mask()[model_indices]
        return mask

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
    ) -> Union[ModelImage, ModelImageList]:
        """Sample the group model on an image. Produces the flux values for
        each pixel associated with the models in this group. Each
        model is called individually and the results are added
        together in one larger image.

        Args:
          image (Optional["Model_Image"]): Image to sample on, overrides the windows for each sub model, they will all be evaluated over this entire image. If left as none then each sub model will be evaluated in its window.

        """
        if window is None:
            image = self.target[self.window].model_image()
        else:
            image = self.target[window].model_image()

        for model in self.models.values():
            if window is None:
                use_window = model.window
            elif isinstance(image, ImageList) and isinstance(model.target, ImageList):
                indices = image.match_indices(model.target)
                if len(indices) == 0:
                    continue
                use_window = WindowList(window_list=list(image.images[i].window for i in indices))
            elif isinstance(image, ImageList) and isinstance(model.target, Image):
                try:
                    image.index(model.target)
                except ValueError:
                    continue
            elif isinstance(image, Image) and isinstance(model.target, ImageList):
                try:
                    model.target.index(image)
                except ValueError:
                    continue
            elif isinstance(image, Image) and isinstance(model.target, Image):
                if image.identity != model.target.identity:
                    continue
                use_window = window
            else:
                raise NotImplementedError(
                    f"Group_Model cannot sample with {type(image)} and {type(model.target)}"
                )
            image += model(window=model.window & use_window)

        return image

    @torch.no_grad()
    def jacobian(
        self,
        pass_jacobian: Optional[JacobianImage] = None,
        window: Optional[Window] = None,
    ) -> JacobianImage:
        """Compute the jacobian for this model. Done by first constructing a
        full jacobian (Npixels * Nparameters) of zeros then call the
        jacobian method of each sub model and add it in to the total.

        Args:
          pass_jacobian (Optional["Jacobian_Image"]): A Jacobian image pre-constructed to be passed along instead of constructing new Jacobians

        """
        if window is None:
            window = self.window

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.build_params_array_identities()
            )
        else:
            jac_img = pass_jacobian

        for model in self.models.values():
            model.jacobian(
                pass_jacobian=jac_img,
                window=window,
            )

        return jac_img

    def __iter__(self):
        return (mod for mod in self.models.values())

    @property
    def target(self) -> Optional[Union[TargetImage, TargetImageList]]:
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar: Optional[Union[TargetImage, TargetImageList]]):
        if not (tar is None or isinstance(tar, (TargetImage, TargetImageList))):
            raise InvalidTarget("Group_Model target must be a Target_Image instance.")
        self._target = tar
