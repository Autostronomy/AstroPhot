from typing import Optional, Union

import numpy as np

from ..backend_obj import backend
from ..param import forward
from ..errors import InvalidTarget, InvalidWindow
from .base import Model
from .model_object import ComponentModel
from ..image import TargetImageBatch, TargetImage, Window, WindowBatch


class BatchModel(Model):
    """A batch of models for an object in an image.

    This is a batch of models for an object in an image. It has a position on the sky
    determined by `center` and may or may not be convolved with a PSF to represent some data.
    """

    usable = True

    def __init__(self, *, model: ComponentModel = None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(
            model, ComponentModel
        ), "BatchModel must be initialized with a ComponentModel instance."
        self.hierarchical_link("model", model)

    @property
    def target(self) -> Optional[Union[TargetImageBatch, TargetImage]]:
        try:
            if self._target is not None:
                return self._target
        except AttributeError:
            pass
        return self.model.target

    @target.setter
    def target(self, target: Optional[Union[TargetImageBatch, TargetImage]]):
        if not (target is None or isinstance(target, (TargetImageBatch, TargetImage))):
            raise InvalidTarget(
                "BatchModel target must be a TargetImageBatch or TargetImage instance."
            )
        try:
            del self._target  # Remove old target if it exists
        except AttributeError:
            pass

        self._target = target

    @property
    def window(self) -> Optional[Window]:
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
            if self.model is None:
                raise ValueError(
                    "This batch model has no model or window, these must be provided by the user"
                )
            return self.model.window
        return self._window

    @window.setter
    def window(self, window):
        if window is None:
            self._window = None
            return
        if isinstance(window, (Window, WindowBatch)):
            self._window = window
            return
        try:
            window = np.array(window)
        except Exception:
            raise InvalidWindow(f"Unrecognized window format: {str(window)}")
        if window.shape == (4,) or window.shape == (2, 2):
            assert isinstance(
                self.target, TargetImage
            ), "Window format (4,) or (2, 2) requires a TargetImage target."
            self._window = Window(window, image=self.target)
        elif (
            window.ndim == 2
            and window.shape[1] == 4
            or window.ndim == 3
            and window.shape[1:] == (2, 2)
        ):
            assert isinstance(
                self.target, TargetImageBatch
            ), "Window batch format requires a TargetImageBatch target."
            self._window = WindowBatch(window, image=self.target)
        else:
            raise InvalidWindow(f"Unrecognized window format: {str(window)}")

    @forward
    def __call__(self, window=None, model_params=None, model_dims=None, **kwargs):

        # Window within which to evaluate model
        if window is None:
            window = self.window
        else:
            window = window & self.window

        batch_img = None if isinstance(self.target, TargetImage) else 0
        working_image = self.target.model_image(window)
        I, J = self.model._pixel_meshgridder(
            working_image, pad=self.model.psf.psf_pad, upsample=self.model.psf.upsample
        )
        # pixel_collecting_area: Units from flux/arcsec^2 to flux, multiply by pixel area
        sample = backend.vmap(
            self.model.sample, in_dims=(batch_img, batch_img, batch_img, None, None, model_dims)
        )(
            I,
            J,
            working_image.pixel_collecting_area,
            self.model.psf.psf_pad,
            self.model.psf.upsample,
            model_params,
        )
        if isinstance(self.target, TargetImage) and isinstance(self.window, Window):
            sample = backend.sum(sample, dim=0)
        working_image._data = sample
        return working_image
