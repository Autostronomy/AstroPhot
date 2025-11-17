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
    """Metropolis-Adjusted Langevin Algorithm (MALA) sampler, based on:
    https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm . This
    is a gradient-based MCMC sampler that uses the gradient of the
    log-likelihood to propose new samples. These gradient based proposals can
    lead to more efficient sampling of the parameter space. This is especially
    true when the mass_matrix is set well. A good guess for the mass matrix is
    the covariance matrix of the likelihood at the maximum likelihood point.
    Which can be found fairly easily with the LM optimizer (see the fitting
    methods tutorial).

    **Args:**
    -  `chains`: The number of MCMC chains to run in parallel. Default is 4.
    -  `epsilon`: The step size for the MALA sampler. Default is 1e-2.
    -  `mass_matrix`: The mass matrix for the MALA sampler. If None, the identity matrix is used.
    -  `progress_bar`: Whether to show a progress bar during sampling. Default is True.
    -  `likelihood`: The likelihood function to use for the MCMC sampling. Can be "gaussian" or "poisson". Default is "gaussian".
    """

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

        self.chain, self.logp = func.mala(
            initial_state,
            Px,
            dPdx,
            self.max_iter,
            self.epsilon,
            self.mass_matrix,
            progress=self.progress_bar,
            desc="MALA",
        )
        # Fill model with max logp sample
        max_logp_index = np.argmax(self.logp)
        max_logp_index = np.unravel_index(max_logp_index, self.logp.shape)
        self.model.fill_dynamic_values(
            backend.as_array(self.chain[max_logp_index], dtype=config.DTYPE, device=config.DEVICE)
        )

        return self
