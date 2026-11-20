import logging
from typing import Sequence, Optional

import numpy as np

from .. import config
from ..backend_obj import backend, ArrayLike
from ..models import Model

__all__ = ("BaseOptimizer",)


class BaseOptimizer:
    """
    Base optimizer object that other optimizers inherit from. Ensures consistent signature for the classes.

    :param model: an AstroPhot_Model object that will have its (unlocked) parameters optimized [AstroPhot_Model]
    :param initial_state: optional initialization for the parameters as a 1D Array [Array]
    :param relative_tolerance: tolerance for counting success steps as: $0 < (\\chi_2^2 - \\chi_1^2)/\\chi_1^2 < \\text{tol}$ [float]
    :param verbose: how much the optimizer reports while it runs. With None, the default, this is left to the AstroPhot logger: progress is logged at the INFO level and extra detail at DEBUG, see ``ap.set_logger``. A number sets it for this optimizer alone: 0 for no progress, 1 for progress, 2 to add the extra detail [int]
    :param max_iter: maximum allowed number of iterations [int]
    :param save_steps: optional string for path to save the model at each step (fitter dependent), e.g. "model_step_{step}.hdf5" [str]
    :param fit_valid: whether to fit while forcing parameters into valid range, or allow any value for each parameter. Default True [bool]

    """

    def __init__(
        self,
        model: Model,
        initial_state: Sequence = None,
        relative_tolerance: float = 1e-3,
        verbose: Optional[int] = None,
        max_iter: int = None,
        save_steps: Optional[str] = None,
        fit_valid: bool = True,
    ) -> None:

        self.model = model
        self.verbose = verbose

        if initial_state is None:
            self.current_state = model.get_values()
        else:
            self.current_state = backend.as_array(
                initial_state, dtype=config.DTYPE, device=config.DEVICE
            )

        self.max_iter = max_iter if max_iter is not None else 100 * len(self.current_state)
        self.iteration = 0
        self.save_steps = save_steps
        self.fit_valid = fit_valid

        self.relative_tolerance = relative_tolerance
        self.lambda_history = []
        self.loss_history = []
        self.message = ""

    def report(self, message: str, detail: int = 1, level: int = logging.INFO) -> None:
        """Send a message about the fit to the AstroPhot logger.

        `detail` is 1 for the progress of the fit and 2 for extra detail. By
        default the level of the logger chooses what is shown, with extra detail
        logged at DEBUG. If `verbose` was given to the optimizer it chooses
        instead, messages up to that `detail` are logged at `level` and the
        rest are dropped.
        """
        if self.verbose is None:
            config.logger.log(level if detail <= 1 else logging.DEBUG, message)
        elif self.verbose >= detail:
            config.logger.log(level, message)

    def fit(self) -> "BaseOptimizer":
        raise NotImplementedError("Please use a subclass of BaseOptimizer for optimization")

    def step(self, current_state: ArrayLike = None) -> None:
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
            return backend.to_numpy(self.current_state)
        return np.array(self.lambda_history)[N][np.argmin(np.array(self.loss_history)[N])]

    def res_loss(self):
        """returns the minimum value from the loss history."""
        N = np.isfinite(self.loss_history)
        return np.min(np.array(self.loss_history)[N])
