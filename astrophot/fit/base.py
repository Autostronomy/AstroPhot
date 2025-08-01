from typing import Sequence, Optional

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammainc

from .. import config
from ..models import Model
from ..image import Window


__all__ = ("BaseOptimizer",)


class BaseOptimizer:
    """
    Base optimizer object that other optimizers inherit from. Ensures consistent signature for the classes.

    **Args:**
    -  `model`: an AstroPhot_Model object that will have its (unlocked) parameters optimized [AstroPhot_Model]
    -  `initial_state`: optional initialization for the parameters as a 1D tensor [tensor]
    -  `relative_tolerance`: tolerance for counting success steps as: $0 < (\\chi_2^2 - \\chi_1^2)/\\chi_1^2 < \\text{tol}$ [float]
    -  `fit_window`: optional window to fit the model on [Window]
    -  `verbose`: verbosity level for the optimizer [int]
    -  `max_iter`: maximum allowed number of iterations [int]
    -  `save_steps`: optional string for path to save the model at each step (fitter dependent), e.g. "model_step_{step}.hdf5" [str]
    -  `fit_valid`: whether to fit while forcing parameters into valid range, or allow any value for each parameter. Default True [bool]

    """

    def __init__(
        self,
        model: Model,
        initial_state: Sequence = None,
        relative_tolerance: float = 1e-3,
        fit_window: Optional[Window] = None,
        verbose: int = 1,
        max_iter: int = None,
        save_steps: Optional[str] = None,
        fit_valid: bool = True,
    ) -> None:

        self.model = model
        self.verbose = verbose

        if initial_state is None:
            self.current_state = model.build_params_array()
        else:
            self.current_state = torch.as_tensor(
                initial_state, dtype=model.dtype, device=model.device
            )

        if fit_window is None:
            self.fit_window = self.model.window
        else:
            self.fit_window = fit_window & self.model.window

        self.max_iter = max_iter if max_iter is not None else 100 * len(self.current_state)
        self.iteration = 0
        self.save_steps = save_steps
        self.fit_valid = fit_valid

        self.relative_tolerance = relative_tolerance
        self.lambda_history = []
        self.loss_history = []
        self.message = ""

    def fit(self) -> "BaseOptimizer":
        raise NotImplementedError("Please use a subclass of BaseOptimizer for optimization")

    def step(self, current_state: torch.Tensor = None) -> None:
        raise NotImplementedError("Please use a subclass of BaseOptimizer for optimization")

    def chi2min(self) -> float:
        """
        Returns the minimum value of chi^2 loss in the loss history.
        """
        return np.nanmin(self.loss_history)

    def res(self) -> np.ndarray:
        """Returns the value of lambda (state parameters) at which minimum loss was achieved."""
        N = np.isfinite(self.loss_history)
        if np.sum(N) == 0:
            config.logger.warning(
                "Getting optimizer res with no real loss history, using current state"
            )
            return self.current_state.detach().cpu().numpy()
        return np.array(self.lambda_history)[N][np.argmin(np.array(self.loss_history)[N])]

    def res_loss(self):
        """returns the minimum value from the loss history."""
        N = np.isfinite(self.loss_history)
        return np.min(np.array(self.loss_history)[N])

    @staticmethod
    def chi2contour(n_params: int, confidence: float = 0.682689492137) -> float:
        """
        Calculates the chi^2 contour for the given number of parameters.

        **Args:**
        - `n_params` (int): The number of parameters.
        - `confidence` (float, optional): The confidence interval (default is 0.682689492137).
        """

        def _f(x: float, nu: int) -> float:
            """Helper function for calculating chi^2 contour."""
            return (gammainc(nu / 2, x / 2) - confidence) ** 2

        for method in ["L-BFGS-B", "Powell", "Nelder-Mead"]:
            res = minimize(_f, x0=n_params, args=(n_params,), method=method, tol=1e-8)

            if res.success:
                return res.x[0]
        raise RuntimeError(f"Unable to compute Chi^2 contour for n params: {n_params}")
