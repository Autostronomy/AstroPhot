from typing import Optional
from ..backend_obj import backend
from ..param import forward
from ..errors import SpecificationConflict
from .base import Model
from .mixins import SampleMixin, GradMixin
from .model_object import ComponentModel
from ..image import TargetImage, Window, TargetImageBatch, WindowBatch
from . import func
from .. import config


class BatchModel(GradMixin, SampleMixin, Model):
    """A batch of models that all share the same window/target.

    This can for example be used to model a crowded area of the sky with many
    overlapping sources, or to model a single object that is represented by many
    components (consider this a generalization of the Multi-gaussian expansion
    model). If you want to model the same object in multiple images, see the
    BatchSceneModel instead.

    **Note:** any model parameters that you wish to batch over must be set to
        dynamic=True. See [caskade hierarchical
        models](https://caskade.readthedocs.io/en/latest/notebooks/HierarchicalModels.html)
        for more details.
    """

    usable = True
    _model_type = "batch"

    def __init__(self, *, model: ComponentModel = None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(
            model, ComponentModel
        ), "BatchModel must be initialized with a ComponentModel instance."
        self.hierarchical_link("model", model)

    def initialize(self):
        self.model.initialize()

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

    @forward
    def __call__(self, model_params=None, model_dims=None, **kwargs):
        psf, upsample, pad = self.model._prep_psf()
        working_image = self.target.model_image(self.window)
        I, J = self.model._pixel_meshgridder(self.target, self.window, pad, upsample)
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


class BatchSceneModel(GradMixin, Model):
    """A single model as viewed in multiple images.

    This model is quite restrictive in its use, but can provide a significant
    speedup by vectorizing the evaluation of the model. Some key things to keep
    in mind:

    - All model parameters that you wish to batch over must be set to
      dynamic=True. See [caskade hierarchical
      models](https://caskade.readthedocs.io/en/latest/notebooks/HierarchicalModels.html)
      for more details.

    - You must use a TargetImageBatch as the target, meaning that all the images
      must be the same number of pixels.

    - You must use a WindowBatch as the window (or none to just use the full
      images), meaning that the windows must all be the same shape in pixels.

    - The model you provide must have a TargetImage target and Window window,
      and these must have the same number of pixels as the batched versions.

    - If the base model has a PSFModel as its PSF then you cannot override it
      (that's part of the model), otherwise the PSF from the TargetImageBatch
      will override the model PSF for all images.
    """

    usable = True
    _model_type = "batch scene"

    def __init__(self, *, model: Model = None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(model.target, TargetImage):
            raise SpecificationConflict(
                f"BatchSceneModel can only be used with models that have a TargetImage as their target, not a {type(model.target).__name__}."
            )
        self.hierarchical_link("model", model)

    def initialize(self):
        self.model.initialize()

    @property
    def target(self) -> Optional[TargetImageBatch]:
        return self._target

    @target.setter
    def target(self, target: TargetImageBatch):
        assert isinstance(
            target, TargetImageBatch
        ), "BatchSceneModel target must be a TargetImageBatch."
        self._target = target

    @property
    def window(self) -> WindowBatch:
        if self._window is None:
            return self._target.window
        return self._window

    @window.setter
    def window(self, window):
        assert window is None or isinstance(
            window, WindowBatch
        ), "BatchSceneModel window must be a WindowBatch."
        self._window = window

    @forward
    def __call__(self, model_params=None, model_dims=None, **kwargs):
        working_image = self.target.model_image(self.window)
        crtan = working_image.crtan
        shift = backend.as_array(
            self.window.origin_shifter(self.model.window), dtype=config.DTYPE, device=config.DEVICE
        )
        crpix = working_image.crpix + shift
        CD = working_image.CD
        psf = self.target.psf_stack
        psf_batch = None if psf is None else 0
        working_image._data = backend.vmap(
            lambda *args: self.model(*args)._data, in_dims=(0, 0, 0, psf_batch, model_dims)
        )(CD, crtan, crpix, psf, model_params)
        return working_image
