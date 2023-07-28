# Apply a different optimizer iteratively
from typing import Dict, Any, Sequence, Union
import os
from time import time
from copy import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import gammainc

from .base import BaseOptimizer
from .lm import LM
from .. import AP_config

__all__ = ["Iter", "Iter_LM"]


class Iter(BaseOptimizer):
    """Optimizer wrapper that performs optimization iteratively.

    This optimizer applies a different optimizer to a group model iteratively.
    It can be used for complex fits or when the number of models to fit is too large to fit in memory.

    Args:
        model: An `AutoPhot_Model` object to perform optimization on.
        method: The optimizer class to apply at each iteration step.
        initial_state: Optional initial state for optimization, defaults to None.
        max_iter: Maximum number of iterations, defaults to 100.
        method_kwargs: Keyword arguments to pass to `method`.
        **kwargs: Additional keyword arguments.

    Attributes:
        ndf: Degrees of freedom of the data.
        method: The optimizer class to apply at each iteration step. Default: Levenberg-Marquardt
        method_kwargs: Keyword arguments to pass to `method`.
        iteration: The number of iterations performed.
        lambda_history: A list of the states at each iteration step.
        loss_history: A list of the losses at each iteration step
    """

    def __init__(
        self,
        model: "AutoPhot_Model",
        method: "BaseOptimizer" = LM,
        initial_state: np.ndarray = None,
        max_iter: int = 100,
        method_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.method = method
        self.method_kwargs = method_kwargs
        if "relative_tolerance" not in method_kwargs and isinstance(method, LM):
            # Lower tolerance since it's not worth fine tuning a model when it's neighbors will be shifting soon anyway
            self.method_kwargs["relative_tolerance"] = 1e-3
            self.method_kwargs["max_iter"] = 15
        #          # pixels      # parameters
        self.ndf = self.model.target[self.model.window].flatten("data").size(0) - len(
            self.current_state
        )
        if self.model.target.has_mask:
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(
                self.model.target[self.model.window].flatten("mask")
            ).item()

    def sub_step(self, model: "AutoPhot_Model") -> None:
        """
        Perform optimization for a single model.

        Args:
            model: The model to perform optimization on.
        """
        self.Y -= model()
        initial_target = model.target
        model.target = model.target[model.window] - self.Y[model.window]
        res = self.method(model, **self.method_kwargs).fit()
        self.Y += model()
        if self.verbose > 1:
            AP_config.ap_logger.info(res.message)
        model.target = initial_target

    def step(self) -> None:
        """
        Perform a single iteration of optimization.
        """
        if self.verbose > 0:
            AP_config.ap_logger.info("--------iter-------")

        # Fit each model individually
        for model in self.model.models.values():
            if self.verbose > 0:
                AP_config.ap_logger.info(model.name)
            self.sub_step(model)
        # Update the current state
        self.current_state = self.model.parameters.get_vector(as_representation=True)

        # Update the loss value
        with torch.no_grad():
            if self.verbose > 0:
                AP_config.ap_logger.info("Update Chi^2 with new parameters")
            self.Y = self.model(
                parameters=self.current_state,
                as_representation=True,
            )
            D = self.model.target[self.model.window].flatten("data")
            V = (
                self.model.target[self.model.window].flatten("variance")
                if self.model.target.has_variance
                else 1.0
            )
            if self.model.target.has_mask:
                M = self.model.target[self.model.window].flatten("mask")
                loss = (
                    torch.sum(
                        (((D - self.Y.flatten("data")) ** 2) / V)[torch.logical_not(M)]
                    )
                    / self.ndf
                )
            else:
                loss = torch.sum(((D - self.Y.flatten("data")) ** 2 / V)) / self.ndf
        if self.verbose > 0:
            AP_config.ap_logger.info(f"Loss: {loss.item()}")
        self.lambda_history.append(np.copy((self.current_state).detach().cpu().numpy()))
        self.loss_history.append(loss.item())

        # Test for convergence
        if self.iteration >= 2 and (
            (-self.relative_tolerance * 1e-3)
            < ((self.loss_history[-2] - self.loss_history[-1]) / self.loss_history[-1])
            < (self.relative_tolerance / 10)
        ):
            self._count_finish += 1
        else:
            self._count_finish = 0

        self.iteration += 1

    def fit(self) -> "BaseOptimizer":
        """
        Fit the models to the target.


        """

        self.iteration = 0
        self.Y = self.model(parameters=self.current_state, as_representation=True)
        start_fit = time()
        try:
            while True:
                self.step()
                if self.save_steps is not None:
                    self.model.save(
                        os.path.join(
                            self.save_steps,
                            f"{self.model.name}_Iteration_{self.iteration:03d}.yaml",
                        )
                    )
                if self.iteration > 2 and self._count_finish >= 2:
                    self.message = self.message + "success"
                    break
                elif self.iteration >= self.max_iter:
                    self.message = (
                        self.message + f"fail max iterations reached: {self.iteration}"
                    )
                    break

        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"

        self.model.parameters.set_values(self.res(), as_representation=True)
        if self.verbose > 1:
            AP_config.ap_logger.info(
                f"Iter Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )

        return self


class Iter_LM(BaseOptimizer):
    """Optimization wrapper that call LM optimizer on subsets of variables.

    Iter_LM takes the full set of parameters for a model and breaks
    them down into chunks as specified by the user. It then calls
    Levenberg-Marquardt optimization on the subset of parameters, and
    iterates through all subsets until every parameter has been
    optimized. It cycles through these chunks until convergence. This
    method is very powerful in situations where the full optimization
    problem cannot fit in memory, or where the optimization problem is
    too complex to tackle as a single large problem. In full LM
    optimization a single problematic parameter can ripple into issues
    with every other parameter, so breaking the problem down can
    sometimes make an otherwise intractable problem easier. For small
    problems with only a few models, it is likely better to optimize
    the full problem with LM as, when it works, LM is faster than the
    Iter_LM method.

    Args:
      chunks (Union[int, tuple]): Specify how to break down the model parameters. If an integer, at each iteration the algorithm will break the parameters into groups of that size. If a tuple, should be a tuple of tuples of strings which give an explicit pairing of parameters to optimize, note that it is allowed to have variable size chunks this way. Default: 50
      method (str): How to iterate through the chunks. Should be one of: random, sequential. Default: random
    """

    def __init__(
        self,
        model: "AutoPhot_Model",
        initial_state: Sequence = None,
        chunks: Union[int, tuple] = 50,
        max_iter: int = 100,
        method: str = "random",
        LM_kwargs: dict = {},
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.chunks = chunks
        self.method = method
        self.LM_kwargs = LM_kwargs

        #          # pixels      # parameters
        self.ndf = self.model.target[self.model.window].flatten("data").numel() - len(
            self.current_state
        )
        if self.model.target.has_mask:
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(
                self.model.target[self.model.window].flatten("mask")
            ).item()

    def step(self):
        # These store the chunking information depending on which chunk mode is selected
        param_ids = list(self.model.parameters.get_identity_vector())
        _chunk_index = 0
        _chunk_choices = None
        res = None

        if self.verbose > 0:
            AP_config.ap_logger.info("--------iter-------")

        # Loop through all the chunks
        while True:
            if isinstance(self.chunks, int):
                if len(param_ids) == 0:
                    break
                if self.method == "random":
                    # Draw a random chunk of ids
                    if len(param_ids) >= self.chunks:
                        chunk = random.sample(param_ids, self.chunks)
                        N = np.argsort(
                            np.array(list(param_ids.index(c) for c in chunk))
                        )
                        chunk = np.array(chunk)[N]
                    else:
                        chunk = copy(param_ids)
                else:
                    # Draw the next chunk of ids
                    chunk = param_ids[: self.chunks]
                # Remove the selected ids from the list
                for p in chunk:
                    param_ids.pop(param_ids.index(p))
            elif isinstance(self.chunks, (tuple, list)):
                if _chunk_choices is None:
                    # Make a list of the chunks as given explicitly
                    _chunk_choices = list(range(len(self.chunks)))
                if self.method == "random":
                    if len(_chunk_choices) == 0:
                        break
                    # Select a random chunk from the given groups
                    sub_index = random.choice(_chunk_choices)
                    _chunk_choices.pop(_chunk_choices.index(sub_index))
                    chunk = self.chunks[sub_index]
                else:
                    if _chunk_index >= len(self.chunks):
                        break
                    # Select the next chunk in order
                    chunk = self.chunks[_chunk_index]
                    _chunk_index += 1
            else:
                raise ValueError(
                    "Unrecognized chunks value, should be one of int, tuple. not: {type(self.chunks)}"
                )
            if self.verbose > 0:
                AP_config.ap_logger.info(str(chunk))
            del res
            res = LM(
                self.model,
                fit_parameters_identity=chunk,
                ndf=self.ndf,
                **self.LM_kwargs,
            ).fit()
            if self.verbose > 0:
                AP_config.ap_logger.info(f"chunk loss: {res.res_loss()}")
            if self.verbose > 1:
                AP_config.ap_logger.info(f"chunk message: {res.message}")

        self.loss_history.append(res.res_loss())
        self.lambda_history.append(
            self.model.parameters.get_vector(as_representation=True)
            .detach()
            .cpu()
            .numpy()
        )
        if self.verbose > 0:
            AP_config.ap_logger.info(f"Loss: {self.loss_history[-1]}")

        # test for convergence
        if self.iteration >= 2 and (
            (-self.relative_tolerance * 1e-3)
            < ((self.loss_history[-2] - self.loss_history[-1]) / self.loss_history[-1])
            < (self.relative_tolerance / 10)
        ):
            self._count_finish += 1
        else:
            self._count_finish = 0

        self.iteration += 1

    def fit(self):
        self.iteration = 0

        start_fit = time()
        try:
            while True:
                self.step()
                if self.save_steps is not None:
                    self.model.save(
                        os.path.join(
                            self.save_steps,
                            f"{self.model.name}_Iteration_{self.iteration:03d}.yaml",
                        )
                    )
                if self.iteration > 2 and self._count_finish >= 2:
                    self.message = self.message + "success"
                    break
                elif self.iteration >= self.max_iter:
                    self.message = (
                        self.message + f"fail max iterations reached: {self.iteration}"
                    )
                    break

        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"

        self.model.parameters.set_values(self.res(), as_representation=True)
        if self.verbose > 1:
            AP_config.ap_logger.info(
                f"Iter Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )

        return self
