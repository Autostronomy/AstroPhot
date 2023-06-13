# Traditional gradient descent with Adam
from time import time
from typing import Sequence
import torch
import numpy as np

from .base import BaseOptimizer
from .. import AP_config

__all__ = ["Grad"]


class Grad(BaseOptimizer):
    """A gradient descent optimization wrapper for AutoPhot_Model objects.

    The default method is "NAdam", a variant of the Adam optimization algorithm.
    This optimizer uses a combination of gradient descent and Nesterov momentum for faster convergence.
    The optimizer is instantiated with a set of initial parameters and optimization options provided by the user.
    The `fit` method performs the optimization, taking a series of gradient steps until a stopping criteria is met.

    Parameters:
        model (AutoPhot_Model): an AutoPhot_Model object with which to perform optimization.
        initial_state (torch.Tensor, optional): an optional initial state for optimization.
        method (str, optional): the optimization method to use for the update step. Defaults to "NAdam".
        patience (int or None, optional): the number of iterations without improvement before the optimizer will exit early. Defaults to None.
        optim_kwargs (dict, optional): a dictionary of keyword arguments to pass to the pytorch optimizer.

    Attributes:
        model (AutoPhot_Model): the AutoPhot_Model object to optimize.
        current_state (torch.Tensor): the current state of the parameters being optimized.
        iteration (int): the number of iterations performed during the optimization.
        loss_history (list): the history of loss values at each iteration of the optimization.
        lambda_history (list): the history of parameter values at each iteration of the optimization.
        optimizer (torch.optimizer): the PyTorch optimizer object being used.
        patience (int or None): the number of iterations without improvement before the optimizer will exit early.
        method (str): the optimization method being used.
        optim_kwargs (dict): the dictionary of keyword arguments passed to the PyTorch optimizer.


    """

    def __init__(
        self, model: "AutoPhot_Model", initial_state: Sequence = None, **kwargs
    ) -> None:
        """Initialize the gradient descent optimizer.

        Args:
            - model: instance of the model to be optimized.
            - initial_state: Initial state of the model.
            - patience: (optional) If a positive integer, then stop the optimization if there has been no improvement in the loss for this number of iterations.
            - method: (optional) The name of the optimization method to use. Default is NAdam.
            - optim_kwargs: (optional) Keyword arguments to be passed to the optimizer.
        """

        super().__init__(model, initial_state, **kwargs)
        self.current_state.requires_grad = True

        # set parameters from the user
        self.patience = kwargs.get("patience", None)
        self.method = kwargs.get("method", "NAdam").strip()
        self.optim_kwargs = kwargs.get("optim_kwargs", {})
        self.report_freq = kwargs.get("report_freq", 10)

        # Default learning rate if none given. Equalt to 1 / sqrt(parames)
        if not "lr" in self.optim_kwargs:
            self.optim_kwargs["lr"] = 0.1 / (len(self.current_state) ** (0.5))

        # Instantiates the appropriate pytorch optimizer with the initial state and user provided kwargs
        self.optimizer = getattr(torch.optim, self.method)(
            (self.current_state,), **self.optim_kwargs
        )

    def compute_loss(self) -> torch.Tensor:
        Ym = self.model(parameters=self.current_state, as_representation=True).flatten(
            "data"
        )
        Yt = self.model.target[self.model.window].flatten("data")
        W = (
            self.model.target[self.model.window].flatten("variance")
            if self.model.target.has_variance
            else 1.0
        )
        ndf = len(Yt) - len(self.current_state)
        if self.model.target.has_mask:
            mask = self.model.target[self.model.window].flatten("mask")
            ndf -= torch.sum(mask)
            mask = torch.logical_not(mask)
            loss = torch.sum((Ym[mask] - Yt[mask]) ** 2 / W[mask]) / ndf
        else:
            loss = torch.sum((Ym - Yt) ** 2 / W) / ndf
        return loss

    def step(self) -> None:
        """Take a single gradient step. Take a single gradient step.

        Computes the loss function of the model,
        computes the gradient of the parameters using automatic differentiation,
        and takes a step with the PyTorch optimizer.

        """

        self.iteration += 1

        self.optimizer.zero_grad()

        loss = self.compute_loss()

        loss.backward()

        self.loss_history.append(loss.detach().cpu().item())
        self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
        if (
            self.iteration % int(self.max_iter / self.report_freq) == 0
        ) or self.iteration == self.max_iter:
            if self.verbose > 0:
                AP_config.ap_logger.info(f"iter: {self.iteration}, loss: {loss.item()}")
            if self.verbose > 1:
                AP_config.ap_logger.info(f"gradient: {self.current_state.grad}")
        self.optimizer.step()

    def fit(self) -> "BaseOptimizer":
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
                    and (len(self.loss_history) - np.argmin(self.loss_history))
                    > self.patience
                ):
                    self.message = self.message + " fail no improvement"
                    break
                L = np.sort(self.loss_history)
                if len(L) >= 3 and 0 < L[1] - L[0] < 1e-6 and 0 < L[2] - L[1] < 1e-6:
                    self.message = self.message + " success"
                    break
        except KeyboardInterrupt:
            self.message = self.message + " fail interrupted"

        # Set the model parameters to the best values from the fit and clear any previous model sampling
        self.model.parameters.set_values(
            torch.tensor(self.res()), as_representation=True
        )
        if self.verbose > 1:
            AP_config.ap_logger.info(
                f"Grad Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )
        return self
