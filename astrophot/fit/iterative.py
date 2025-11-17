# Apply a different optimizer iteratively
from typing import Dict, Any, Union, Sequence, Literal
from time import time
from functools import partial

from caskade import ValidContext
import numpy as np
import torch

from .base import BaseOptimizer
from ..models import Model
from .lm import LM
from .. import config
from ..backend_obj import backend
from ..errors import OptimizeStopSuccess, OptimizeStopFail
from . import func

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
            V = self.model.target[self.model.window].flatten("variance")
            M = self.model.target[self.model.window].flatten("mask")
            loss = backend.sum((((D - self.Y.flatten("data")) ** 2) / V)[~M]) / self.ndf
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


class IterParam(BaseOptimizer):
    """Optimization wrapper that call LM optimizer on subsets of variables.

    IterParam takes the full set of parameters for a model and breaks them down
    into chunks as specified by the user. It then calls Levenberg-Marquardt
    optimization on the subset of parameters, and iterates through all subsets
    until every parameter has been optimized. It cycles through these chunks
    until convergence. This method is very powerful in situations where the full
    optimization problem cannot fit in memory, or where the optimization problem
    is too complex to tackle as a single large problem. In full LM optimization
    a single problematic parameter can ripple into issues with every other
    parameter, so breaking the problem down can sometimes make an otherwise
    intractable problem easier. For small problems with only a few models, it is
    likely better to optimize the full problem with LM as, when it works, LM is
    faster than the IterParam method.

    Args:
      chunks (Union[int, tuple]): Specify how to break down the model
        parameters. If an integer, at each iteration the algorithm will break the
        parameters into groups of that size. If a tuple, should be a tuple of
        arrays of length num_dimensions which act as selectors for the parameters
        to fit (1 to include, 0 to exclude). Default: 50
      chunk_order (str): How to iterate through the chunks. Should be one of: random,
        sequential. Default: sequential
    """

    def __init__(
        self,
        model: Model,
        initial_state: Sequence = None,
        chunks: Union[int, tuple] = 50,
        chunk_order: Literal["random", "sequential"] = "sequential",
        max_iter: int = 100,
        relative_tolerance: float = 1e-5,
        Lup=11.0,
        Ldn=9.0,
        L0=1.0,
        max_step_iter: int = 10,
        ndf=None,
        W=None,
        likelihood="gaussian",
        **kwargs,
    ):

        super().__init__(
            model,
            initial_state,
            max_iter=max_iter,
            relative_tolerance=relative_tolerance,
            **kwargs,
        )
        # Maximum number of iterations of the algorithm
        self.max_iter = max_iter
        # Maximum number of steps while searching for chi^2 improvement on a single jacobian evaluation
        self.max_step_iter = max_step_iter
        self.Lup = Lup
        self.Ldn = Ldn
        self.L = L0
        self.likelihood = likelihood
        if self.likelihood not in ["gaussian", "poisson"]:
            raise ValueError(f"Unsupported likelihood: {self.likelihood}")
        self.chunks = self.make_chunks(chunks)
        self.chunk_order = chunk_order

        # mask
        fit_mask = self.model.fit_mask()
        if isinstance(fit_mask, tuple):
            fit_mask = backend.concatenate(tuple(FM.flatten() for FM in fit_mask))
        else:
            fit_mask = fit_mask.flatten()

        mask = self.model.target[self.fit_window].flatten("mask")
        mask = mask | fit_mask
        self.mask = ~mask

        if backend.sum(self.mask).item() == 0:
            raise OptimizeStopSuccess("No data to fit. All pixels are masked")

        # Initialize optimizer attributes
        self.Y = self.model.target[self.fit_window].flatten("data")[self.mask]

        # 1 / (sigma^2)
        if W is not None:
            self.W = backend.as_array(W, dtype=config.DTYPE, device=config.DEVICE).flatten()[
                self.mask
            ]
        else:
            self.W = self.model.target[self.fit_window].flatten("weight")[self.mask]

        # The forward model which computes the output image given input parameters
        self.full_forward = lambda x: model(window=self.fit_window, params=x).flatten("data")[
            self.mask
        ]
        self.forward = []
        # Compute the jacobian
        self.jacobian = []

        f = lambda c, state, x: model(
            window=self.fit_window,
            params=backend.fill_at_indices(backend.copy(state), self.chunks[c], x),
        ).flatten("data")[self.mask]
        j = backend.jacfwd(
            lambda c, state, x: self.model(
                window=self.fit_window,
                params=backend.fill_at_indices(backend.copy(state), self.chunks[c], x),
            ).flatten("data")[self.mask],
            argnums=2,
        )
        for c in range(len(self.chunks)):
            self.forward.append(partial(f, c))
            self.jacobian.append(partial(j, c))

        # variable to store covariance matrix if it is ever computed
        self._covariance_matrix = None

        # Degrees of freedom
        if ndf is None:
            self.ndf = max(1.0, len(self.Y) - len(self.current_state))
        else:
            self.ndf = ndf

    def make_chunks(self, chunks):
        if isinstance(chunks, int):
            new_chunks = []
            for i in range(0, len(self.current_state), chunks):
                chunk = np.zeros(len(self.current_state), dtype=bool)
                chunk[i : i + chunks] = True
                new_chunks.append(chunk)
            chunks = new_chunks
        return chunks

    def iter_chunks(self):
        if self.chunk_order == "random":
            chunk_ids = list(range(len(self.chunks)))
            np.random.shuffle(chunk_ids)
        elif self.chunk_order == "sequential":
            chunk_ids = list(range(len(self.chunks)))
        else:
            raise ValueError(
                f"Unrecognized chunk_order: {self.chunk_order}. Should be one of: random, sequential"
            )
        return chunk_ids

    def chi2_ndf(self):
        return (
            backend.sum(self.W * (self.Y - self.full_forward(self.current_state)) ** 2) / self.ndf
        )

    def poisson_2nll_ndf(self):
        M = self.full_forward(self.current_state)
        return 2 * backend.sum(M - self.Y * backend.log(M + 1e-10)) / self.ndf

    @torch.no_grad()
    def fit(self, update_uncertainty=True) -> BaseOptimizer:
        """This performs the fitting operation. It iterates the LM step
        function until convergence is reached. Includes a message
        after fitting to indicate how the fitting exited. Typically if
        the message returns a "success" then the algorithm found a
        minimum. This may be the desired solution, or a pathological
        local minimum, this often depends on the initial conditions.

        """

        if len(self.current_state) == 0:
            if self.verbose > 0:
                config.logger.warning("No parameters to optimize. Exiting fit")
            self.message = "No parameters to optimize. Exiting fit"
            return self

        if self.likelihood == "gaussian":
            quantity = "Chi^2/DoF"
            self.loss_history = [self.chi2_ndf().item()]
        elif self.likelihood == "poisson":
            quantity = "2NLL/DoF"
            self.loss_history = [self.poisson_2nll_ndf().item()]
        self._covariance_matrix = None
        self.L_history = [self.L]
        self.lambda_history = [backend.to_numpy(backend.copy(self.current_state))]
        if self.verbose > 0:
            config.logger.info(
                f"==Starting LM fit for '{self.model.name}' with {len(self.current_state)} dynamic parameters and {len(self.Y)} pixels=="
            )

        for _ in range(self.max_iter):
            # Report status
            if self.verbose > 0:
                config.logger.info(f"{quantity}: {self.loss_history[-1]:.6g}, L: {self.L:.3g}")

            # Perform fitting
            chunk_L = []
            for c in self.iter_chunks():
                try:
                    if self.fit_valid:
                        with ValidContext(self.model):
                            valid_state = self.model.to_valid(self.current_state)
                            res = func.lm_step(
                                x=valid_state[self.chunks[c]],
                                data=self.Y,
                                model=partial(self.forward[c], valid_state),
                                weight=self.W,
                                jacobian=partial(self.jacobian[c], valid_state),
                                L=self.L,
                                Lup=self.Lup,
                                Ldn=self.Ldn,
                                likelihood=self.likelihood,
                            )
                        self.current_state = self.model.from_valid(
                            backend.fill_at_indices(
                                valid_state, self.chunks[c], backend.copy(res["x"])
                            )
                        )
                    else:
                        res = func.lm_step(
                            x=self.current_state[self.chunks[c]],
                            data=self.Y,
                            model=partial(self.forward[c], self.current_state),
                            weight=self.W,
                            jacobian=partial(self.jacobian[c], self.current_state),
                            L=self.L,
                            Lup=self.Lup,
                            Ldn=self.Ldn,
                            likelihood=self.likelihood,
                        )
                        self.current_state = backend.fill_at_indices(
                            self.current_state, self.chunks[c], backend.copy(res["x"])
                        )
                except OptimizeStopFail:
                    if self.verbose > 0:
                        config.logger.warning(
                            f"Could not find step to improve Chi^2 on chunk {c}, moving to next chunk"
                        )
                    continue
                except OptimizeStopSuccess as e:
                    continue  # success on individual chunk is not enough to stop overall fit
                chunk_L.append(res["L"])

            # Record progress
            self.L = np.clip(np.max(chunk_L), 1e-9, 1e9)
            self.L_history.append(self.L)
            self.loss_history.append(2 * res["nll"] / self.ndf)
            self.lambda_history.append(backend.to_numpy(backend.copy(self.current_state)))
            if self.check_convergence():
                break

        else:
            self.message = self.message + "fail. Maximum iterations"

        if self.verbose > 0:
            config.logger.info(
                f"Final {quantity}: {np.nanmin(self.loss_history):.6g}, L: {self.L_history[np.nanargmin(self.loss_history)]:.3g}. Converged: {self.message}"
            )

        self.model.fill_dynamic_values(
            backend.as_array(self.res(), dtype=config.DTYPE, device=config.DEVICE)
        )
        if update_uncertainty:
            self.update_uncertainty()

        return self

    def check_convergence(self) -> bool:
        """Check if the optimization has converged based on the last
        iteration's chi^2 and the relative tolerance.
        """
        if len(self.loss_history) < 3:
            return False
        good_history = [self.loss_history[0]]
        for l in self.loss_history[1:]:
            if good_history[-1] > l:
                good_history.append(l)
        if len(self.loss_history) - len(good_history) >= 10:
            self.message = self.message + "success by immobility. Convergence not guaranteed"
            return True
        if len(good_history) < 3:
            return False
        if (good_history[-2] - good_history[-1]) / good_history[
            -1
        ] < self.relative_tolerance and self.L < 0.1:
            self.message = self.message + "success"
            return True
        if len(good_history) < 10:
            return False
        if (good_history[-10] - good_history[-1]) / good_history[-1] < self.relative_tolerance:
            self.message = self.message + "success by immobility. Convergence not guaranteed"
            return True
        return False

    @property
    @torch.no_grad()
    def covariance_matrix(self):
        """The covariance matrix for the model at the current
        parameters. This can be used to construct a full Gaussian PDF for the
        parameters using: $\\mathcal{N}(\\mu,\\Sigma)$ where $\\mu$ is the
        optimized parameters and $\\Sigma$ is the covariance matrix.

        """

        if self._covariance_matrix is not None:
            return self._covariance_matrix

        N = len(self.current_state)
        self._covariance_matrix = backend.zeros((N, N), dtype=config.DTYPE, device=config.DEVICE)
        for c in self.iter_chunks():
            J = self.jacobian[c](self.current_state, self.current_state[self.chunks[c]])
            if self.likelihood == "gaussian":
                hess = func.hessian(J, self.W)
            elif self.likelihood == "poisson":
                hess = func.hessian_poisson(J, self.Y, self.full_forward(self.current_state))
            try:
                sub_covariance_matrix = backend.linalg.inv(hess)
            except:
                config.logger.warning(
                    "WARNING: Hessian is singular, likely at least one parameter is non-physical. Will use pseudo-inverse of Hessian to continue but results should be inspected."
                )
                sub_covariance_matrix = backend.linalg.pinv(hess)

            ids = backend.meshgrid(
                backend.as_array(np.arange(N)[self.chunks[c]], dtype=int, device=config.DEVICE),
                backend.as_array(np.arange(N)[self.chunks[c]], dtype=int, device=config.DEVICE),
                indexing="ij",
            )
            self._covariance_matrix = backend.fill_at_indices(
                self._covariance_matrix, (ids[0], ids[1]), sub_covariance_matrix
            )
        return self._covariance_matrix

    @torch.no_grad()
    def update_uncertainty(self) -> None:
        """Call this function after optimization to set the uncertainties for
        the parameters. This will use the diagonal of the covariance
        matrix to update the uncertainties. See the covariance_matrix
        function for the full representation of the uncertainties.

        """
        # set the uncertainty for each parameter
        cov = self.covariance_matrix
        if backend.all(backend.isfinite(cov)):
            try:
                self.model.fill_dynamic_value_uncertainties(
                    backend.sqrt(backend.abs(backend.diag(cov)))
                )
            except RuntimeError as e:
                config.logger.warning(f"Unable to update uncertainty due to: {e}")
        else:
            config.logger.warning(
                "Unable to update uncertainty due to non finite covariance matrix"
            )
