from typing import Optional
from ..backend_obj import backend
from ..param import forward
from .base import Model
from .mixins import SampleMixin
from .model_object import ComponentModel
from ..image import TargetImage, Window
from . import func


class BatchModel(SampleMixin, Model):
    """A batch of models that all share the same window/target.

    This can for example be used to model a crowded area of the sky with many
    overlapping sources, or to model a single object that is represented by many
    components (consider this a generalization of the Multi-gaussian expansion
    model).
    """

    usable = True
    _model_type = "batch"

    def __init__(self, *, model: ComponentModel = None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(
            model, ComponentModel
        ), "BatchModel must be initialized with a ComponentModel instance."
        self.hierarchical_link("model", model)

    @property
    def target(self) -> Optional[TargetImage]:
        return self.model.target

    @target.setter
    def target(self, target: Optional[TargetImage]):
        pass

    @property
    def window(self) -> Optional[Window]:
        """The window defines a region on the sky in which this model will be
        optimized and evaluated. Two models with non-overlapping windows are in
        effect independent of each other. If there is another model with a
        window that spans both of them, then they are tenuously connected.

        If not provided, the model will assume a window equal to the target it
        is fitting. Note that in this case the window is not explicitly set to
        the target window, so if the model is moved to another target then the
        fitting window will also change.

        """
        return self.model.window

    @window.setter
    def window(self, window):
        pass

    @property
    def mask(self):
        return self.model.mask

    @mask.setter
    def mask(self, mask):
        pass

    def fit_mask(self):
        return self.model.fit_mask()

    @forward
    def __call__(self, window=None, model_params=None, model_dims=None, **kwargs):

        # Window within which to evaluate model
        if window is None:
            window = self.window
        else:
            window = window & self.window

        psf, upsample, pad = self.model._prep_psf()
        working_image = self.target.model_image(window)
        I, J = self.model._pixel_meshgridder(self.target, window, pad, upsample)
        Z = backend.vmap(
            self.model.sample,
            in_dims=(None, None, None, None, None, model_dims),
        )(
            I,
            J,
            None,
            pad,
            upsample,
            model_params,
        )
        Z = backend.sum(Z, dim=0)
        if psf is not None and not self.model.internal_psf:
            if isinstance(psf, Model):
                psf = psf()._data
            if psf.shape != (1, 1):  # skip if identity PSF
                Z = func.convolve(Z, psf)
                Z = Z[pad : Z.shape[0] - pad, pad : Z.shape[1] - pad]
                Z = func.downsample(Z, upsample)
        working_image._data = Z
        return working_image
