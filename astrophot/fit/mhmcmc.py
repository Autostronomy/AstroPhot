# Metropolis-Hasting Markov-Chain Monte-Carlo
from typing import Optional, Sequence

import numpy as np

try:
    import emcee
except ImportError:
    emcee = None

from .base import BaseOptimizer
from ..models import Model
from .. import config
from ..backend_obj import backend

__all__ = ["MHMCMC"]


class MHMCMC(BaseOptimizer):
    """Metropolis-Hastings Markov-Chain Monte-Carlo sampler, based on:
    https://en.wikipedia.org/wiki/Metropolis-Hastings_algorithm . This is simply
    a thin wrapper for the Emcee package, which is a well-known MCMC sampler.

    Note that the Emcee sampler requires multiple walkers to sample the
    parameter space efficiently. The number of walkers is set to twice the
    number of parameters by default, but can be made higher (not lower) if desired.
    This is done by passing a 2D array of shape (nwalkers, ndim) to the `fit` method.

    **Args:**
    -  `likelihood`: The likelihood function to use for the MCMC sampling. Can be "gaussian" or "poisson". Default is "gaussian".
    """

    def __init__(
        self,
        model: Model,
        initial_state: Optional[Sequence] = None,
        max_iter: int = 1000,
        likelihood="gaussian",
        **kwargs,
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        if emcee is None:
            raise ImportError(
                "The emcee package is required for MHMCMC sampling. Please install it with `pip install emcee` or the like."
            )
        self.likelihood = likelihood

        self.chain = []

    def density(self, state: np.ndarray) -> np.ndarray:
        """
        Returns the density of the model at the given state vector.
        This is used to calculate the likelihood of the model at the given state.
        """
        state = backend.as_array(state, dtype=config.DTYPE, device=config.DEVICE)
        if self.likelihood == "gaussian":
            return np.array(list(self.model.gaussian_log_likelihood(s).item() for s in state))
        elif self.likelihood == "poisson":
            return np.array(list(self.model.poisson_log_likelihood(s).item() for s in state))
        else:
            raise ValueError(f"Unknown likelihood type: {self.likelihood}")

    def fit(
        self,
        state: Optional[np.ndarray] = None,
        nsamples: Optional[int] = None,
        restart_chain: bool = True,
        skip_initial_state_check: bool = True,
        flat_chain: bool = True,
    ):
        """
        Performs the MCMC sampling using a Metropolis Hastings acceptance step and records the chain for later examination.
        """

        if nsamples is None:
            nsamples = self.max_iter

        if state is None:
            state = self.current_state

        if len(state.shape) == 1:
            nwalkers = state.shape[0] * 2
            state = state * np.random.normal(loc=1, scale=0.01, size=(nwalkers, state.shape[0]))
        else:
            nwalkers = state.shape[0]
        ndim = state.shape[1]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.density, vectorize=True)
        state = sampler.run_mcmc(state, nsamples, skip_initial_state_check=skip_initial_state_check)
        if restart_chain:
            self.chain = sampler.get_chain(flat=flat_chain)
        else:
            self.chain = np.append(self.chain, sampler.get_chain(flat=flat_chain), axis=0)
        self.model.fill_dynamic_values(
            backend.as_array(self.chain[-1], dtype=config.DTYPE, device=config.DEVICE)
        )
        return self
