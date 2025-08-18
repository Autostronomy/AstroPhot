# Traditional gradient descent with Adam
from time import time
from typing import Sequence
from caskade import ValidContext
import torch
import numpy as np

from .base import BaseOptimizer
from .. import config
from ..backend_obj import backend, ArrayLike
from ..models import Model
from ..errors import OptimizeStopFail, OptimizeStopSuccess
from . import func
from ..utils.decorators import combine_docstrings

__all__ = ["Grad"]


@combine_docstrings
class Grad(BaseOptimizer):
    """A gradient descent optimization wrapper for AstroPhot Model objects.

    The default method is "NAdam", a variant of the Adam optimization algorithm.
    This optimizer uses a combination of gradient descent and Nesterov momentum for faster convergence.
    The optimizer is instantiated with a set of initial parameters and optimization options provided by the user.
    The `fit` method performs the optimization, taking a series of gradient steps until a stopping criteria is met.

    **Args:**
    -  `likelihood` (str, optional): The likelihood function to use for the optimization. Defaults to "gaussian".
    -  `method` (str, optional): the optimization method to use for the update step. Defaults to "NAdam".
    -  `optim_kwargs` (dict, optional): a dictionary of keyword arguments to pass to the pytorch optimizer.
    -  `patience` (int, optional): number of steps with no improvement before stopping the optimization. Defaults to 10.
    -  `report_freq` (int, optional): frequency of reporting the optimization progress. Defaults to 10 steps.
    """

    def __init__(
        self,
        model: Model,
        initial_state: Sequence = None,
        likelihood="gaussian",
        method="NAdam",
        optim_kwargs={},
        patience: int = 10,
        report_freq=10,
        **kwargs,
    ) -> None:

        super().__init__(model, initial_state, **kwargs)

        self.likelihood = likelihood

        # set parameters from the user
        self.patience = patience
        self.method = method
        self.optim_kwargs = optim_kwargs
        self.report_freq = report_freq

        # Default learning rate if none given. Equal to 1 / sqrt(parames)
        if "lr" not in self.optim_kwargs:
            self.optim_kwargs["lr"] = 0.1 / (len(self.current_state) ** (0.5))

        # Instantiates the appropriate pytorch optimizer with the initial state and user provided kwargs
        self.current_state.requires_grad = True
        self.optimizer = getattr(torch.optim, self.method)(
            (self.current_state,), **self.optim_kwargs
        )

    def density(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the density of the model at the given state vector. This is used
        to calculate the likelihood of the model at the given state. Based on
        ``self.likelihood``, will be either the Gaussian or Poisson negative log
        likelihood.
        """
        if self.likelihood == "gaussian":
            return -self.model.gaussian_log_likelihood(state)
        elif self.likelihood == "poisson":
            return -self.model.poisson_log_likelihood(state)
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

    def step(self) -> None:
        """Take a single gradient step.

        Computes the loss function of the model,
        computes the gradient of the parameters using automatic differentiation,
        and takes a step with the PyTorch optimizer.

        """
        self.iteration += 1

        self.optimizer.zero_grad()
        self.current_state.requires_grad = True
        loss = self.density(self.current_state)

        loss.backward()

        self.loss_history.append(backend.to_numpy(loss))
        self.lambda_history.append(np.copy(backend.to_numpy(self.current_state)))
        if (
            self.iteration % int(self.max_iter / self.report_freq) == 0
        ) or self.iteration == self.max_iter:
            if self.verbose > 0:
                config.logger.info(f"iter: {self.iteration}, posterior density: {loss.item():.6e}")
            if self.verbose > 1:
                config.logger.info(f"gradient: {self.current_state.grad}")
        self.optimizer.step()

    def fit(self) -> BaseOptimizer:
        """
        Perform an iterative fit of the model parameters using the specified optimizer.

        The fit procedure continues until a stopping criteria is met,
        such as the maximum number of iterations being reached,
        or no improvement being made after a specified number of iterations.

        """
        start_fit = time()
        try:
            while True:
                self.step()
                if self.iteration >= self.max_iter:
                    self.message = self.message + " fail max iteration reached"
                    break
                if (
                    self.patience is not None
                    and (len(self.loss_history) - np.argmin(self.loss_history)) > self.patience
                ):
                    self.message = self.message + " fail no improvement"
                    break
                L = np.sort(self.loss_history)
                if len(L) >= 5 and 0 < (L[4] - L[0]) / L[0] < self.relative_tolerance:
                    self.message = self.message + " success"
                    break
        except KeyboardInterrupt:
            self.message = self.message + " fail interrupted"

        # Set the model parameters to the best values from the fit and clear any previous model sampling
        self.model.fill_dynamic_values(
            torch.tensor(self.res(), dtype=config.DTYPE, device=config.DEVICE)
        )
        if self.verbose > 1:
            config.logger.info(
                f"Grad Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )
        return self


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

    **Args:**
    -  `S` (float, optional): The initial step size for the Slalom optimizer. Defaults to 1e-4.
    -  `likelihood` (str, optional): The likelihood function to use for the optimization. Defaults to "gaussian".
    -  `report_freq` (int, optional): Frequency of reporting the optimization progress. Defaults to 10 steps.
    -  `relative_tolerance` (float, optional): The relative tolerance for convergence. Defaults to 1e-4.
    -  `momentum` (float, optional): The momentum factor for the Slalom optimizer. Defaults to 0.5.
    -  `max_iter` (int, optional): The maximum number of iterations for the optimizer. Defaults to 1000.
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
        self.model.fill_dynamic_values(
            backend.as_array(self.res(), dtype=config.DTYPE, device=config.DEVICE)
        )
        if self.verbose > 0:
            config.logger.info(
                f"Slalom Fitting complete in {time() - self.start_fit} sec with message: {self.message}"
            )
        return self
