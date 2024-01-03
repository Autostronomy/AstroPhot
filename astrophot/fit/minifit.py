# Apply an optimizer toa  downsampled version of an image
from typing import Dict, Any

import numpy as np

from .base import BaseOptimizer
from ..models import AstroPhot_Model
from .lm import LM
from .. import AP_config

__all__ = ["MiniFit"]


class MiniFit(BaseOptimizer):
    def __init__(
        self,
        model: AstroPhot_Model,
        downsample_factor: int = 1,
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
            if small_target.size < self.max_pixels:
                break
            self.downsample_factor += 1

        if self.verbose > 0:
            AP_config.ap_logger.info(f"Downsampling target by {self.downsample_factor}x")

        self.small_target = small_target
        self.model.target = small_target
        res = self.method(self.model, **self.method_kwargs).fit()
        self.model.target = initial_target

        self.message = res.message

        return self
