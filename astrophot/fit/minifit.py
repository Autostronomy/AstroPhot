# Apply an optimizer toa  downsampled version of an image
from typing import Dict, Any

import numpy as np

from .base import BaseOptimizer
from ..models import Model
from .lm import LM
from .. import config

__all__ = ["MiniFit"]


class MiniFit(BaseOptimizer):
    """MiniFit optimizer that applies a fitting method to a downsampled version
    of the model's target image.

    This is useful for quickly optimizing parameters on a smaller scale before
    applying them to the full resolution image. With fewer pixels, the optimization
    can be faster and more efficient, especially for large images.

    This Optimizer can wrap any optimizer that follows the BaseOptimizer interface.

    **Args:**
    -  `downsample_factor`: Factor by which to downsample the target image. Default is 2.
    -  `max_pixels`: Maximum number of pixels in the downsampled image. Default is 10000.
    -  `method`: The optimizer method to use, e.g., `LM` for Levenberg-Marquardt. Default is `LM`.
    -  `method_kwargs`: Additional keyword arguments to pass to the optimizer method.
    """

    def __init__(
        self,
        model: Model,
        downsample_factor: int = 2,
        max_pixels: int = 10000,
        method: BaseOptimizer = LM,
        initial_state: np.ndarray = None,
        method_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(model, initial_state, **kwargs)

        self.method = method
        self.method_kwargs = method_kwargs
        if "verbose" not in self.method_kwargs:
            self.method_kwargs["verbose"] = self.verbose

        self.downsample_factor = downsample_factor
        self.max_pixels = max_pixels

    def fit(self) -> BaseOptimizer:
        initial_target = self.model.target
        target_area = self.model.target[self.model.window]
        while True:
            small_target = target_area.reduce(self.downsample_factor)
            if np.prod(small_target.shape) < self.max_pixels:
                break
            self.downsample_factor += 1

        if self.verbose > 0:
            config.logger.info(f"Downsampling target by {self.downsample_factor}x")

        self.small_target = small_target
        self.model.target = small_target
        res = self.method(self.model, **self.method_kwargs).fit()
        self.model.target = initial_target

        self.message = res.message

        return self
