# Traditional gradient descent with Adam
from time import time
from typing import Sequence
from caskade import ValidContext

from .base import BaseOptimizer
from .. import config
from ..backend_obj import backend, ArrayLike
from ..models import Model
from ..errors import OptimizeStopFail, OptimizeStopSuccess
from . import func

__all__ = ["Grad"]


class Slalom(BaseOptimizer):
    """Slalom optimizer for Model objects.

    Slalom is a gradient descent optimization algorithm that uses a few
    evaluations along the direction of the gradient to find the optimal step
    size. This is done by assuming that the posterior density is a parabola and
    then finding the minimum.

    The optimizer quickly finds the minimum of the posterior density along the
    gradient direction, then updates the gradient at the new position and
    repeats. This continues until it reaches a set of 5 steps which collectively
    improve the posterior density by an amount smaller than the
    `relative_tolerance` threshold, indicating that convergence has been
    achieved. Note that this convergence criteria is not a guarantee, simply a
    heuristic. The default tolerance was such that the optimizer will
    substantially improve from the starting point, and do so quickly, but may
    not reach all the way to the minimum of the posterior density. Like other
    gradient descent algorithms, Slalom slows down considerably when trying to
    achieve very high precision.

    :param S: The initial step size for the Slalom optimizer. Defaults to 1e-4.
    :type S: float, optional
    :param likelihood: The likelihood function to use for the optimization. Defaults to "gaussian".
    :type likelihood: str, optional
    :param report_freq: Frequency of reporting the optimization progress. Defaults to 10 steps.
    :type report_freq: int, optional
    :param relative_tolerance: The relative tolerance for convergence. Defaults to 1e-4.
    :type relative_tolerance: float, optional
    :param momentum: The momentum factor for the Slalom optimizer. Defaults to 0.5.
    :type momentum: float, optional
    :param max_iter: The maximum number of iterations for the optimizer. Defaults to 1000.
    :type max_iter: int, optional
    """

    def __init__(
        self,
        model: Model,
        initial_state: Sequence = None,
        S=1e-4,
        likelihood: str = "gaussian",
        report_freq: int = 10,
        relative_tolerance: float = 1e-4,
        momentum: float = 0.5,
        max_iter: int = 1000,
        **kwargs,
    ) -> None:
        """Initialize the Slalom optimizer."""
        super().__init__(
            model, initial_state, relative_tolerance=relative_tolerance, max_iter=max_iter, **kwargs
        )
        self.likelihood = likelihood
        self.S = S
        self.report_freq = report_freq
        self.momentum = momentum

    def density(self, state: ArrayLike) -> ArrayLike:
        """Calculate the density of the model at the given state. Based on
        ``self.likelihood``, will be either the Gaussian or Poisson negative log
        likelihood."""
        if self.likelihood == "gaussian":
            return -self.model.gaussian_log_likelihood(state)
        elif self.likelihood == "poisson":
            return -self.model.poisson_log_likelihood(state)
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

    def fit(self) -> BaseOptimizer:
        """Perform the Slalom optimization."""

        grad_func = backend.grad(self.density)
        momentum = backend.zeros_like(self.current_state)
        self.S_history = [self.S]
        self.loss_history = [self.density(self.current_state).item()]
        self.lambda_history = [backend.to_numpy(self.current_state)]
        self.start_fit = time()

        for i in range(self.max_iter):

            try:
                # Perform the Slalom step
                vstate = self.model.to_valid(self.current_state)
                with ValidContext(self.model):
                    self.S, loss, grad = func.slalom_step(
                        self.density, grad_func, vstate, m=momentum, S=self.S
                    )
                self.current_state = self.model.from_valid(
                    vstate - self.S * (grad + momentum) / backend.linalg.norm(grad + momentum)
                )
                momentum = self.momentum * (momentum + grad)
            except OptimizeStopSuccess as e:
                self.message = self.message + str(e)
                break
            except OptimizeStopFail as e:
                if backend.allclose(momentum, backend.zeros_like(momentum)):
                    self.message = self.message + str(e)
                    break
                momentum = backend.zeros_like(self.current_state)
                continue
            # Log the loss
            self.S_history.append(self.S)
            self.loss_history.append(loss)
            self.lambda_history.append(backend.to_numpy(self.current_state))

            if self.verbose > 0 and (i % int(self.report_freq) == 0 or i == self.max_iter - 1):
                config.logger.info(
                    f"iter: {i}, step size: {self.S:.6e}, posterior density: {loss:.6e}"
                )

            if len(self.loss_history) >= 5:
                relative_loss = (self.loss_history[-5] - self.loss_history[-1]) / self.loss_history[
                    -1
                ]
                if relative_loss < self.relative_tolerance:
                    self.message = self.message + " success"
                    break
        else:
            self.message = self.message + " fail. max iteration reached"

        # Set the model parameters to the best values from the fit
        self.model.set_values(
            backend.as_array(self.res(), dtype=config.DTYPE, device=config.DEVICE)
        )
        if self.verbose > 0:
            config.logger.info(
                f"Slalom Fitting complete in {time() - self.start_fit} sec with message: {self.message}"
            )
        return self
