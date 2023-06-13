from time import time
from typing import Any, Union, Sequence, Optional

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammainc

from .. import AP_config


__all__ = ["BaseOptimizer"]


class BaseOptimizer(object):
    """
    Base optimizer object that other optimizers inherit from. Ensures consistent signature for the classes.

    Parameters:
        model: an AutoPhot_Model object that will have it's (unlocked) parameters optimized [AutoPhot_Model]
        initial_state: optional initialization for the parameters as a 1D tensor [tensor]
        max_iter: maximum allowed number of iterations [int]
        relative_tolerance: tolerance for counting success steps as: 0 < (Chi2^2 - Chi1^2)/Chi1^2 < tol [float]

    """

    def __init__(
        self,
        model: "Autorof_Model",
        initial_state: Sequence = None,
        relative_tolerance: float = 1e-3,
        fit_parameters_identity: Optional[tuple] = None,
        fit_window: Optional["Window"] = None,
        **kwargs,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            model (object): An object representing the model.
            initial_state (Optional[Sequence]): The initial state of the model could be any tensor.
                           If `None`, the model's default initial state will be used.
            relative_tolerance (float): The relative tolerance for the optimization.
            fit_parameters_identity (Optiona[tuple]): a tuple of parameter identity strings which tell the LM optimizer which parameters of the model to fit.
            **kwargs (dict): Additional keyword arguments.

        Attributes:
            model (object): An object representing the model.
            verbose (int): The verbosity level.
            current_state (Tensor): The current state of the model.
            max_iter (int): The maximum number of iterations.
            iteration (int): The current iteration number.
            save_steps (Optional[str]): Save intermediate results to this path.
            relative_tolerance (float): The relative tolerance for the optimization.
            lambda_history (List[ndarray]): A list of the optimization steps.
            loss_history (List[float]): A list of the optimization losses.
            message (str): An informational message.
        """

        self.model = model
        self.verbose = kwargs.get("verbose", 0)
        self.fit_parameters_identity = fit_parameters_identity

        if fit_window is None:
            self.fit_window = self.model.window
        else:
            self.fit_window = fit_window & self.model.window

        if initial_state is None:
            self.model.initialize()
            initial_state = self.model.parameters.get_vector(
                as_representation=True,
                parameters_identity=self.fit_parameters_identity,
            )
        else:
            initial_state = torch.as_tensor(
                initial_state, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

        self.current_state = torch.as_tensor(
            initial_state, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if self.verbose > 1:
            AP_config.ap_logger.info(f"initial state: {self.current_state}")
        self.max_iter = kwargs.get("max_iter", 100 * len(initial_state))
        self.iteration = 0
        self.save_steps = kwargs.get("save_steps", None)

        self.relative_tolerance = relative_tolerance
        self.lambda_history = []
        self.loss_history = []
        self.message = ""

    def fit(self) -> "BaseOptimizer":
        """
        Raises:
            NotImplementedError: Error is raised if this method is not implemented in a subclass of BaseOptimizer.
        """
        raise NotImplementedError(
            "Please use a subclass of BaseOptimizer for optimization"
        )

    def step(self, current_state: torch.Tensor = None) -> None:
        """Args:
            current_state (torch.Tensor, optional): Current state of the model parameters. Defaults to None.

        Raises:
            NotImplementedError: Error is raised if this method is not implemented in a subclass of BaseOptimizer.
        """
        raise NotImplementedError(
            "Please use a subclass of BaseOptimizer for optimization"
        )

    def chi2min(self) -> float:
        """
        Returns the minimum value of chi^2 loss in the loss history.

        Returns:
                float: Minimum value of chi^2 loss.
        """
        return np.nanmin(self.loss_history)

    def res(self) -> np.ndarray:
        """Returns the value of lambda (regularization strength) at which minimum chi^2 loss was achieved.

        Returns: ndarray which is the Value of lambda at which minimum chi^2 loss was achieved.
        """
        N = np.isfinite(self.loss_history)
        if np.sum(N) == 0:
            AP_config.ap_logger.warn(
                "Getting optimizer res with no real loss history, using current state"
            )
            return self.current_state.detach().cpu().numpy()
        return np.array(self.lambda_history)[N][
            np.argmin(np.array(self.loss_history)[N])
        ]

    def res_loss(self):
        N = np.isfinite(self.loss_history)
        return np.min(np.array(self.loss_history)[N])

    @staticmethod
    def chi2contour(n_params: int, confidence: float = 0.682689492137) -> float:
        """
        Calculates the chi^2 contour for the given number of parameters.

        Args:
            n_params (int): The number of parameters.
            confidence (float, optional): The confidence interval (default is 0.682689492137).

        Returns:
            float: The calculated chi^2 contour value.

        Raises:
            RuntimeError: If unable to compute the Chi^2 contour for the given number of parameters.

        """

        def _f(x: float, nu: int) -> float:
            """Helper function for calculating chi^2 contour."""
            return (gammainc(nu / 2, x / 2) - confidence) ** 2

        for method in ["L-BFGS-B", "Powell", "Nelder-Mead"]:
            res = minimize(_f, x0=n_params, args=(n_params,), method=method, tol=1e-8)

            if res.success:
                return res.x[0]
        raise RuntimeError(f"Unable to compute Chi^2 contour for ndf: {ndf}")
