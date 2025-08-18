from typing import Sequence, Literal

from scipy.optimize import minimize
import numpy as np

from .base import BaseOptimizer
from .. import config
from ..backend_obj import backend

__all__ = ("ScipyFit",)


class ScipyFit(BaseOptimizer):
    """Scipy-based optimizer for fitting models to data using various
    optimization methods.

    The optimizer uses the `scipy.optimize.minimize` function to perform the
    fitting. The Scipy package is widely used and well tested for optimization
    tasks. It supports a variety of methods, however only a subset allow users to
    define boundaries for the parameters. This wrapper is only for those methods.

    **Args:**
    -  `model`: The model to fit, which should be an instance of `Model`.
    -  `initial_state`: Initial guess for the model parameters as a 1D tensor.
    -  `method`: The optimization method to use. Default is "Nelder-Mead", but can be set to any of: "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", or "trust-constr".
    -  `ndf`: Optional number of degrees of freedom for the fit. If not provided, it is calculated as the number of data points minus the number of parameters.
    """

    def __init__(
        self,
        model,
        initial_state: Sequence = None,
        method: Literal[
            "Nelder-Mead", "L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr"
        ] = "Nelder-Mead",
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
        ndf=None,
        **kwargs,
    ):

        super().__init__(model, initial_state, **kwargs)
        self.method = method
        self.likelihood = likelihood

        # Degrees of freedom
        if ndf is None:
            sub_target = self.model.target[self.model.window]
            ndf = (
                np.prod(sub_target.flatten("data").shape)
                - backend.sum(sub_target.flatten("mask")).item()
            )
            self.ndf = max(1.0, ndf - len(self.current_state))
        else:
            self.ndf = ndf

    def numpy_bounds(self):
        """Convert the model's parameter bounds to a format suitable for scipy.optimize."""
        bounds = []
        for param in self.model.dynamic_params:
            if param.shape == ():
                bound = [None, None]
                if param.valid[0] is not None:
                    bound[0] = backend.to_numpy(param.valid[0])
                if param.valid[1] is not None:
                    bound[1] = backend.to_numpy(param.valid[1])
                bounds.append(tuple(bound))
            else:
                for i in range(np.prod(param.value.shape)):
                    bound = [None, None]
                    if param.valid[0] is not None:
                        bound[0] = backend.to_numpy(param.valid[0].flatten()[i])
                    if param.valid[1] is not None:
                        bound[1] = backend.to_numpy(param.valid[1].flatten()[i])
                    bounds.append(tuple(bound))
        return bounds

    def density(self, state: Sequence) -> float:
        if self.likelihood == "gaussian":
            return -self.model.gaussian_log_likelihood(
                backend.as_array(state, dtype=config.DTYPE, device=config.DEVICE)
            ).item()
        elif self.likelihood == "poisson":
            return -self.model.poisson_log_likelihood(
                backend.as_array(state, dtype=config.DTYPE, device=config.DEVICE)
            ).item()
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

    def fit(self):

        res = minimize(
            lambda x: self.density(x),
            self.current_state,
            method=self.method,
            bounds=self.numpy_bounds(),
            options={
                "maxiter": self.max_iter,
            },
        )
        self.scipy_res = res
        self.message = self.message + f"success: {res.success}, message: {res.message}"
        self.current_state = backend.as_array(res.x, dtype=config.DTYPE, device=config.DEVICE)
        if self.verbose > 0:
            config.logger.info(
                f"Final 2NLL/DoF: {2*self.density(res.x)/self.ndf:.6g}. Converged: {self.message}"
            )
        self.model.fill_dynamic_values(self.current_state)

        return self
