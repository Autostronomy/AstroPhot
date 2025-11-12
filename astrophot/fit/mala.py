# Metropolis-Adjusted Langevin Algorithm sampler
from typing import Optional, Sequence

import numpy as np

from .base import BaseOptimizer
from ..models import Model
from .. import config
from ..backend_obj import backend
from . import func

__all__ = ("MALA",)


class MALA(BaseOptimizer):
    def __init__(
        self,
        model: Model,
        initial_state: Optional[Sequence] = None,
        chains=4,
        epsilon: float = 1e-2,
        mass_matrix: Optional[np.ndarray] = None,
        max_iter: int = 1000,
        progress_bar: bool = True,
        likelihood="gaussian",
        **kwargs,
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)
        self.chain = []
        if len(self.current_state.shape) == 2:
            self.chains = self.current_state.shape[0]
        else:
            self.chains = chains
        self.likelihood = likelihood
        self.epsilon = epsilon
        self.mass_matrix = mass_matrix
        self.progress_bar = progress_bar

    def density_func(self):
        """
        Returns the density of the model at the given state vector.
        This is used to calculate the likelihood of the model at the given state.
        """
        if self.likelihood == "gaussian":
            vll = backend.vmap(self.model.gaussian_log_likelihood)
        elif self.likelihood == "poisson":
            vll = backend.vmap(self.model.poisson_log_likelihood)
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

        def dens(state: np.ndarray) -> np.ndarray:
            state = backend.as_array(state, dtype=config.DTYPE, device=config.DEVICE)
            return backend.to_numpy(vll(state))

        return dens

    def density_grad_func(self):
        """
        Returns the gradient of the density of the model at the given state vector.
        This is used to calculate the gradient of the likelihood of the model at the given state.
        """
        if self.likelihood == "gaussian":
            vll_grad = backend.vmap(backend.grad(self.model.gaussian_log_likelihood))
        elif self.likelihood == "poisson":
            vll_grad = backend.vmap(backend.grad(self.model.poisson_log_likelihood))
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

        def grad(state: np.ndarray) -> np.ndarray:
            state = backend.as_array(state, dtype=config.DTYPE, device=config.DEVICE)
            return backend.to_numpy(vll_grad(state))

        return grad

    def fit(self):

        Px = self.density_func()
        dPdx = self.density_grad_func()

        initial_state = backend.to_numpy(self.current_state)
        if len(initial_state.shape) == 1:
            initial_state = np.repeat(initial_state[None, :], self.chains, axis=0)

        if self.mass_matrix is None:
            D = initial_state.shape[1]
            self.mass_matrix = np.eye(D, dtype=initial_state.dtype)

        self.chain = func.mala(
            initial_state,
            Px,
            dPdx,
            self.max_iter,
            self.epsilon,
            self.mass_matrix,
            progress=self.progress_bar,
            desc="MALA",
        )

        return self.chain
