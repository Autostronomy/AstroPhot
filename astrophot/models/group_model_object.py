from typing import Optional, Sequence, Union

import torch
import numpy as np
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
    JacobianImageList,
)
from .. import config
from ..backend_obj import backend, ArrayLike
from ..utils.decorators import ignore_numpy_warnings
from ..errors import InvalidTarget, InvalidWindow

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
        for model in models:
            if not isinstance(model, Model):
                raise TypeError(f"Expected a Model instance in 'models', got {type(model)}")
        self.models = models
        self._update_window()

    def _update_window(self):
        """Makes a new window object which encloses all the windows of the
        sub models in this group model object.

        """
        if isinstance(self.target, ImageList):  # WindowList if target is a TargetImageList
            new_window = list(target.window.copy() for target in self.target)
            n_windows = [0] * len(self.target.images)
            for model in self.models:
                if isinstance(model.target, ImageList):
                    for target, window in zip(model.target, model.window):
                        index = self.target.index(target)
                        if n_windows[index] == 0:
                            new_window[index] &= window
                        else:
                            new_window[index] |= window
                        n_windows[index] += 1
                elif isinstance(model.target, TargetImage):
                    index = self.target.index(model.target)
                    if n_windows[index] == 0:
                        new_window[index] &= model.window
                    else:
                        new_window[index] |= model.window
                    n_windows[index] += 1
                else:
                    raise NotImplementedError(
                        f"Group_Model cannot construct a window for itself using {type(model.target)} object. Must be a Target_Image"
                    )
            new_window = WindowList(new_window)
            for i, n in enumerate(n_windows):
                if n == 0:
                    config.logger.warning(
                        f"Model {self.name} has no sub models in target '{self.target.images[i].name}', this may cause issues with fitting."
                    )
        else:
            new_window = None
            for model in self.models:
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
        """
        for model in self.models:
            config.logger.info(f"Initializing model {model.name}")
            model.initialize()

    def _fit_mask(self) -> torch.Tensor:
        """Returns a mask for the target image which is the combination of all
        the fit masks of the sub models. This mask is used when the multiple
        models in the group model do not completely overlap with each other, thus
        there are some pixels which are not covered by any model and have no
        reason to be fit.

        """
        subtarget = self.target[self.window]
        if isinstance(subtarget, ImageList):
            mask = list(backend.ones_like(submask) for submask in subtarget._mask)
            for model in self.models:
                model_subtarget = model.target[model.window]
                model_fit_mask = model._fit_mask()
                if isinstance(model_subtarget, ImageList):
                    for target, submask in zip(model_subtarget, model_fit_mask):
                        index = subtarget.index(target)
                        group_indices = subtarget.images[index].get_indices(target.window)
                        model_indices = target.get_indices(subtarget.images[index].window)
                        mask[index] = backend.and_at_indices(
                            mask[index], group_indices, submask[model_indices]
                        )
                else:
                    index = subtarget.index(model_subtarget)
                    group_indices = subtarget.images[index].get_indices(model_subtarget.window)
                    model_indices = model_subtarget.get_indices(subtarget.images[index].window)
                    mask[index] = backend.and_at_indices(
                        mask[index], group_indices, model_fit_mask[model_indices]
                    )
            mask = tuple(mask)
        else:
            mask = backend.ones_like(subtarget._mask)
            for model in self.models:
                model_subtarget = model.target[model.window]
                group_indices = subtarget.get_indices(model.window)
                model_indices = model_subtarget.get_indices(subtarget.window)
                mask = backend.and_at_indices(mask, group_indices, model._fit_mask()[model_indices])
        return mask

    def fit_mask(self) -> torch.Tensor:
        mask = self._fit_mask()
        if isinstance(mask, tuple):
            return tuple(backend.transpose(m, 1, 0) for m in mask)
        return backend.transpose(mask, 1, 0)

    def match_window(self, image: Union[Image, ImageList], window: Window, model: Model) -> Window:
        if isinstance(image, ImageList) and isinstance(model.target, ImageList):
            indices = image.match_indices(model.target)
            if len(indices) == 0:
                raise IndexError
            use_window = WindowList(windows=list(image.images[i].window for i in indices))
        elif isinstance(image, ImageList) and isinstance(model.target, Image):
            try:
                image.index(model.target)
            except ValueError:
                raise IndexError
            use_window = model.window
        elif isinstance(image, Image) and isinstance(model.target, ImageList):
            try:
                i = model.target.index(image)
            except ValueError:
                raise IndexError
            use_window = model.window[i]
        elif isinstance(image, Image) and isinstance(model.target, Image):
            if image.identity != model.target.identity:
                raise IndexError
            use_window = window
        else:
            raise NotImplementedError(
                f"Group_Model cannot sample with {type(image)} and {type(model.target)}"
            )
        return use_window

    def _ensure_vmap_compatible(
        self, image: Union[Image, ImageList], other: Union[Image, ImageList]
    ):
        if isinstance(image, ImageList):
            for img in image.images:
                self._ensure_vmap_compatible(img, other)
            return
        if isinstance(other, ImageList):
            for img in other.images:
                self._ensure_vmap_compatible(image, img)
            return
        if image.identity == other.identity:
            image += backend.zeros_like(other._data[0, 0])

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
    ) -> Union[ModelImage, ModelImageList]:
        """Sample the group model on an image. Produces the flux values for
        each pixel associated with the models in this group. Each
        model is called individually and the results are added
        together in one larger image.

        **Args:**
        -  `image` (Optional[ModelImage]): Image to sample on, overrides the windows for each sub model, they will all be evaluated over this entire image. If left as none then each sub model will be evaluated in its window.

        """
        if window is None:
            image = self.target[self.window].model_image()
        else:
            image = self.target[window].model_image()

        for model in self.models:
            if window is None:
                use_window = model.window
            else:
                try:
                    use_window = self.match_window(image, window, model)
                except IndexError:
                    # If the model target is not in the image, skip it
                    continue
            model_image = model(window=model.window & use_window)
            self._ensure_vmap_compatible(image, model_image)
            image += model_image

        return image

    @torch.no_grad()
    def jacobian(
        self,
        pass_jacobian: Optional[Union[JacobianImage, JacobianImageList]] = None,
        window: Optional[Union[Window, WindowList]] = None,
        params=None,
    ) -> JacobianImage:
        """Compute the jacobian for this model. Done by first constructing a
        full jacobian (Npixels * Nparameters) of zeros then call the
        jacobian method of each sub model and add it in to the total.

        **Args:**
        -  `pass_jacobian` (Optional[JacobianImage]): A Jacobian image pre-constructed to be passed along instead of constructing new Jacobians
        -  `window` (Optional[Window]): A window within which to evaluate the jacobian. If not provided, the model's window will be used.
        -  `params` (Optional[Sequence[Param]]): Parameters to use for the jacobian. If not provided, the model's parameters will be used.

        """
        if window is None:
            window = self.window

        if params is not None:
            self.fill_dynamic_values(params)

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.build_params_array_identities()
            )
        else:
            jac_img = pass_jacobian

        for model in reversed(self.models):
            try:
                use_window = self.match_window(jac_img, window, model)
            except IndexError:
                # If the model target is not in the image, skip it
                continue
            jac_img = model.jacobian(
                pass_jacobian=jac_img,
                window=use_window & model.window,
            )

        return jac_img

    def __iter__(self):
        return (mod for mod in self.models)

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
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = tar

    @property
    def window(self) -> Optional[Union[Window, WindowList]]:
        """The window defines a region on the sky in which this model will be
        optimized and typically evaluated. Two models with
        non-overlapping windows are in effect independent of each
        other. If there is another model with a window that spans both
        of them, then they are tenuously connected.

        If not provided, the model will assume a window equal to the
        target it is fitting. Note that in this case the window is not
        explicitly set to the target window, so if the model is moved
        to another target then the fitting window will also change.

        """
        if self._window is None:
            if self.target is None:
                raise ValueError(
                    "This model has no target or window, these must be provided by the user"
                )
            return self.target.window
        return self._window

    @window.setter
    def window(self, window):
        if window is None:
            self._window = None
        elif isinstance(window, (Window, WindowList)):
            self._window = window
        elif len(window) in [2, 4]:
            self._window = Window(window, image=self.target)
        else:
            raise InvalidWindow(f"Unrecognized window format: {str(window)}")

    def segmentation_map(self) -> ArrayLike:
        """Generate a segmentation map for this group model. Each pixel in the
        segmentation map is assigned an integer value corresponding to the index
        of the sub-model that corresponds to that pixel. The pixels are assigned
        based on "relative importance", meaning that for each pixel, the
        sub-model which contributes the largest fraction of its own total flux to that
        pixel is assigned to it.

        Returns:
            ArrayLike: Segmentation map with the same shape as the target image as windowed by the group model window.

        """
        subtarget = self.target[self.window]
        if isinstance(subtarget, ImageList):
            raise NotImplementedError(
                "Segmentation maps are not currently supported for ImageList targets. Please apply one target at a time."
            )
        else:
            seg_map = backend.zeros_like(subtarget._data, dtype=backend.int32) - 1
            max_flux_frac = (
                0.0 * backend.ones_like(subtarget._data) / np.prod(subtarget._data.shape)
            )
            for idx, model in enumerate(self.models):
                model_image = model()
                model_flux_frac = backend.abs(model_image._data) / backend.sum(
                    backend.abs(model_image._data)
                )
                indices = subtarget.get_indices(model.window)
                model_flux_frac_full = backend.zeros_like(subtarget._data)
                model_flux_frac_full = backend.fill_at_indices(
                    model_flux_frac_full, indices, model_flux_frac
                )
                update_mask = model_flux_frac_full >= max_flux_frac
                seg_map = backend.where(update_mask, idx, seg_map)
                max_flux_frac = backend.where(update_mask, model_flux_frac_full, max_flux_frac)
            return seg_map.T

    def deblend(self) -> Sequence[TargetImage]:
        """Generate deblended images for each sub-model in this group model.
        Each deblended image contains for each pixel, the fraction of the total
        flux at that pixel which is contributed by that sub-model.

        Returns:
            Sequence[TargetImage]: List of deblended TargetImage objects for each sub-model.

        """
        deblended_images = []
        subtarget = self.target[self.window]
        full_model = self()
        if isinstance(subtarget, ImageList):
            raise NotImplementedError(
                "Deblending is not currently supported for ImageList targets. Please apply one target at a time."
            )
        else:
            for model in self.models:
                model_image = model()
                subfull_model = full_model[model.window]
                subsubtarget = subtarget[model.window].copy(
                    name=f"deblend_{model.name}_{subtarget.name}"
                )
                deblend_data = subsubtarget.data * model_image.data / subfull_model.data
                deblend_variance = subsubtarget.variance * model_image.data / subfull_model.data
                subsubtarget.data = deblend_data
                subsubtarget.variance = deblend_variance
                deblended_images.append(subsubtarget)
        return deblended_images
