# Apply a different optimizer iteratively
from typing import Dict, Any
from time import time

import numpy as np
import torch

from .base import BaseOptimizer
from ..models import Model
from .lm import LM
from .. import config
from ..backend_obj import backend

__all__ = [
    "Iter",
    # "Iter_LM"
]


class Iter(BaseOptimizer):
    """Optimizer wrapper that performs optimization iteratively.

    This optimizer applies the LM optimizer to a group model iteratively one
    model at a time. It can be used for complex fits or when the number of
    models to fit is too large to fit in memory. Note that it will iterate
    through the group model, but if models within the group are themselves group
    models, then they will be optimized as a whole. This gives some flexibility
    to structure the models in a useful way.

    If not given, the `lm_kwargs` will be set to a relative tolerance of 1e-3
    and a maximum of 15 iterations. This is to allow for faster convergence, it
    is not worthwhile for a single model to spend lots of time optimizing when
    its neighbors havent converged.

    **Args:**
    -    `max_iter`: Maximum number of iterations, defaults to 100.
    -    `lm_kwargs`: Keyword arguments to pass to `LM` optimizer.
    """

    def __init__(
        self,
        model: Model,
        initial_state: np.ndarray = None,
        max_iter: int = 100,
        lm_kwargs: Dict[str, Any] = {"verbose": 0},
        **kwargs: Dict[str, Any],
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.current_state = model.build_params_array()
        self.lm_kwargs = lm_kwargs
        if "relative_tolerance" not in lm_kwargs:
            # Lower tolerance since it's not worth fine tuning a model when its neighbors will be shifting soon anyway
            self.lm_kwargs["relative_tolerance"] = 1e-3
            self.lm_kwargs["max_iter"] = 15
        #          # pixels      # parameters
        self.ndf = self.model.target[self.model.window].flatten("data").shape[0] - len(
            self.current_state
        )
        if self.model.target.has_mask:
            # subtract masked pixels from degrees of freedom
            self.ndf -= backend.sum(self.model.target[self.model.window].flatten("mask")).item()

    def sub_step(self, model: Model, update_uncertainty=False):
        """
        Perform optimization for a single model.
        """
        self.Y -= model()
        initial_values = model.target.copy()
        model.target = model.target - self.Y
        res = LM(model, **self.lm_kwargs).fit(update_uncertainty=update_uncertainty)
        self.Y += model()
        if self.verbose > 1:
            config.logger.info(res.message)
        model.target = initial_values

    def step(self):
        """
        Perform a single iteration of optimization.
        """
        if self.verbose > 0:
            config.logger.info("--------iter-------")

        # Fit each model individually
        for model in self.model.models:
            if self.verbose > 0:
                config.logger.info(model.name)
            self.sub_step(model)
        # Update the current state
        self.current_state = self.model.build_params_array()

        # Update the loss value
        with torch.no_grad():
            if self.verbose > 0:
                config.logger.info("Update Chi^2 with new parameters")
            self.Y = self.model(params=self.current_state)
            D = self.model.target[self.model.window].flatten("data")
            V = (
                self.model.target[self.model.window].flatten("variance")
                if self.model.target.has_variance
                else 1.0
            )
            if self.model.target.has_mask:
                M = self.model.target[self.model.window].flatten("mask")
                loss = backend.sum((((D - self.Y.flatten("data")) ** 2) / V)[~M]) / self.ndf
            else:
                loss = backend.sum(((D - self.Y.flatten("data")) ** 2 / V)) / self.ndf
        if self.verbose > 0:
            config.logger.info(f"Loss: {loss.item()}")
        self.lambda_history.append(np.copy(backend.to_numpy(self.current_state)))
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

    def fit(self) -> BaseOptimizer:
        """
        Perform the iterative fitting process until convergence or maximum iterations reached.
        """
        self.iteration = 0
        self.Y = self.model(params=self.current_state)
        start_fit = time()
        try:
            while True:
                self.step()
                if self.iteration > 2 and self._count_finish >= 2:
                    self.message = self.message + "success"
                    break
                elif self.iteration >= self.max_iter:
                    self.message = self.message + f"fail max iterations reached: {self.iteration}"
                    break

        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"

        self.model.fill_dynamic_values(
            backend.as_array(self.res(), dtype=config.DTYPE, device=config.DEVICE)
        )
        if self.verbose > 1:
            config.logger.info(
                f"Iter Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )

        return self


# class IterParam(BaseOptimizer):
#     """Optimization wrapper that call LM optimizer on subsets of variables.

#     IterParam takes the full set of parameters for a model and breaks
#     them down into chunks as specified by the user. It then calls
#     Levenberg-Marquardt optimization on the subset of parameters, and
#     iterates through all subsets until every parameter has been
#     optimized. It cycles through these chunks until convergence. This
#     method is very powerful in situations where the full optimization
#     problem cannot fit in memory, or where the optimization problem is
#     too complex to tackle as a single large problem. In full LM
#     optimization a single problematic parameter can ripple into issues
#     with every other parameter, so breaking the problem down can
#     sometimes make an otherwise intractable problem easier. For small
#     problems with only a few models, it is likely better to optimize
#     the full problem with LM as, when it works, LM is faster than the
#     IterParam method.

#     Args:
#       chunks (Union[int, tuple]): Specify how to break down the model parameters. If an integer, at each iteration the algorithm will break the parameters into groups of that size. If a tuple, should be a tuple of tuples of strings which give an explicit pairing of parameters to optimize, note that it is allowed to have variable size chunks this way. Default: 50
#       method (str): How to iterate through the chunks. Should be one of: random, sequential. Default: random
#     """

#     def __init__(
#         self,
#         model: Model,
#         initial_state: Sequence = None,
#         chunks: Union[int, tuple] = 50,
#         max_iter: int = 100,
#         method: str = "random",
#         LM_kwargs: dict = {},
#         **kwargs: Dict[str, Any],
#     ) -> None:
#         super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

#         self.chunks = chunks
#         self.method = method
#         self.LM_kwargs = LM_kwargs

#         #          # pixels      # parameters
#         self.ndf = self.model.target[self.model.window].flatten("data").numel() - len(
#             self.current_state
#         )
#         if self.model.target.has_mask:
#             # subtract masked pixels from degrees of freedom
#             self.ndf -= torch.sum(self.model.target[self.model.window].flatten("mask")).item()

#     def step(self):
#         # These store the chunking information depending on which chunk mode is selected
#         param_ids = list(self.model.parameters.vector_identities())
#         init_param_ids = list(self.model.parameters.vector_identities())
#         _chunk_index = 0
#         _chunk_choices = None
#         res = None

#         if self.verbose > 0:
#             config.logger.info("--------iter-------")

#         # Loop through all the chunks
#         while True:
#             chunk = torch.zeros(len(init_param_ids), dtype=torch.bool, device=config.DEVICE)
#             if isinstance(self.chunks, int):
#                 if len(param_ids) == 0:
#                     break
#                 if self.method == "random":
#                     # Draw a random chunk of ids
#                     for pid in random.sample(param_ids, min(len(param_ids), self.chunks)):
#                         chunk[init_param_ids.index(pid)] = True
#                 else:
#                     # Draw the next chunk of ids
#                     for pid in param_ids[: self.chunks]:
#                         chunk[init_param_ids.index(pid)] = True
#                 # Remove the selected ids from the list
#                 for p in np.array(init_param_ids)[chunk.detach().cpu().numpy()]:
#                     param_ids.pop(param_ids.index(p))
#             elif isinstance(self.chunks, (tuple, list)):
#                 if _chunk_choices is None:
#                     # Make a list of the chunks as given explicitly
#                     _chunk_choices = list(range(len(self.chunks)))
#                 if self.method == "random":
#                     if len(_chunk_choices) == 0:
#                         break
#                     # Select a random chunk from the given groups
#                     sub_index = random.choice(_chunk_choices)
#                     _chunk_choices.pop(_chunk_choices.index(sub_index))
#                     for pid in self.chunks[sub_index]:
#                         chunk[param_ids.index(pid)] = True
#                 else:
#                     if _chunk_index >= len(self.chunks):
#                         break
#                     # Select the next chunk in order
#                     for pid in self.chunks[_chunk_index]:
#                         chunk[param_ids.index(pid)] = True
#                     _chunk_index += 1
#             else:
#                 raise ValueError(
#                     "Unrecognized chunks value, should be one of int, tuple. not: {type(self.chunks)}"
#                 )
#             if self.verbose > 1:
#                 config.logger.info(str(chunk))
#             del res
#             with Param_Mask(self.model.parameters, chunk):
#                 res = LM(
#                     self.model,
#                     ndf=self.ndf,
#                     **self.LM_kwargs,
#                 ).fit()
#             if self.verbose > 0:
#                 config.logger.info(f"chunk loss: {res.res_loss()}")
#             if self.verbose > 1:
#                 config.logger.info(f"chunk message: {res.message}")

#         self.loss_history.append(res.res_loss())
#         self.lambda_history.append(
#             self.model.parameters.vector_representation().detach().cpu().numpy()
#         )
#         if self.verbose > 0:
#             config.logger.info(f"Loss: {self.loss_history[-1]}")

#         # test for convergence
#         if self.iteration >= 2 and (
#             (-self.relative_tolerance * 1e-3)
#             < ((self.loss_history[-2] - self.loss_history[-1]) / self.loss_history[-1])
#             < (self.relative_tolerance / 10)
#         ):
#             self._count_finish += 1
#         else:
#             self._count_finish = 0

#         self.iteration += 1

#     def fit(self):
#         self.iteration = 0

#         start_fit = time()
#         try:
#             while True:
#                 self.step()
#                 if self.save_steps is not None:
#                     self.model.save(
#                         os.path.join(
#                             self.save_steps,
#                             f"{self.model.name}_Iteration_{self.iteration:03d}.yaml",
#                         )
#                     )
#                 if self.iteration > 2 and self._count_finish >= 2:
#                     self.message = self.message + "success"
#                     break
#                 elif self.iteration >= self.max_iter:
#                     self.message = self.message + f"fail max iterations reached: {self.iteration}"
#                     break

#         except KeyboardInterrupt:
#             self.message = self.message + "fail interrupted"

#         self.model.parameters.vector_set_representation(self.res())
#         if self.verbose > 1:
#             config.logger.info(
#                 f"Iter Fitting complete in {time() - start_fit} sec with message: {self.message}"
#             )

#         return self
