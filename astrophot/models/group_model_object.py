from typing import Optional, Sequence

import torch
from caskade import forward

from .core_model import Model
from ..image import (
    Image,
    Target_Image,
    Target_Image_List,
    Image_List,
    Window,
    Window_List,
    Jacobian_Image,
)
from ..utils.decorators import ignore_numpy_warnings
from ..errors import InvalidTarget

__all__ = ["Group_Model"]


class Group_Model(Model):
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
        super().__init__(name=name, models=models, **kwargs)
        self.models = models
        self.update_window()
        if "filename" in kwargs:
            self.load(kwargs["filename"], new_name=name)

    def update_window(self):
        """Makes a new window object which encloses all the windows of the
        sub models in this group model object.

        """
        if isinstance(self.target, Image_List):  # Window_List if target is a Target_Image_List
            new_window = [None] * len(self.target.images)
            for model in self.models.values():
                if isinstance(model.target, Image_List):
                    for target, window in zip(model.target, model.window):
                        index = self.target.index(target)
                        if new_window[index] is None:
                            new_window[index] = window.copy()
                        else:
                            new_window[index] |= window
                elif isinstance(model.target, Target_Image):
                    index = self.target.index(model.target)
                    if new_window[index] is None:
                        new_window[index] = model.window.copy()
                    else:
                        new_window[index] |= model.window
                else:
                    raise NotImplementedError(
                        f"Group_Model cannot construct a window for itself using {type(model.target)} object. Must be a Target_Image"
                    )
            new_window = Window_List(new_window)
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
    def initialize(self, **kwargs):
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
        if isinstance(self.target, Image_List):
            mask = tuple(torch.ones_like(submask) for submask in self.target[self.window].mask)
            for model in self.models.values():
                model_flat_mask = model.fit_mask()
                if isinstance(model.target, Image_List):
                    for target, window, submask in zip(model.target, model.window, model_flat_mask):
                        index = self.target.index(target)
                        group_indices = self.window.window_list[index].get_self_indices(window)
                        model_indices = window.get_self_indices(self.window.window_list[index])
                        mask[index][group_indices] &= submask[model_indices]
                else:
                    index = self.target.index(model.target)
                    group_indices = self.window.window_list[index].get_self_indices(model.window)
                    model_indices = model.window.get_self_indices(self.window.window_list[index])
                    mask[index][group_indices] &= model_flat_mask[model_indices]
        else:
            mask = torch.ones_like(self.target[self.window].mask)
            for model in self.models.values():
                group_indices = self.window.get_self_indices(model.window)
                model_indices = model.window.get_self_indices(self.window)
                mask[group_indices] &= model.fit_mask()[model_indices]
        return mask

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
    ):
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
            elif isinstance(image, Image_List) and isinstance(model.target, Image_List):
                indices = image.match_indices(model.target)
                if len(indices) == 0:
                    continue
                use_window = Window_List(window_list=list(image.images[i].window for i in indices))
            elif isinstance(image, Image_List) and isinstance(model.target, Image):
                try:
                    image.index(model.target)
                except ValueError:
                    continue
            elif isinstance(image, Image) and isinstance(model.target, Image_List):
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
        pass_jacobian: Optional[Jacobian_Image] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
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
                parameters=self.parameters.vector_identities()
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
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        if not (tar is None or isinstance(tar, (Target_Image, Target_Image_List))):
            raise InvalidTarget("Group_Model target must be a Target_Image instance.")
        self._target = tar
