# No U-Turn Sampler variant of Hamiltonian Monte-Carlo
import os
from time import time
from typing import Optional, Sequence
import warnings

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC as pyro_MCMC
from pyro.infer import NUTS as pyro_NUTS
from pyro.infer.mcmc.adaptation import WarmupAdapter, BlockMassMatrix
from pyro.ops.welford import WelfordCovariance

from .base import BaseOptimizer
from .. import AP_config

__all__ = ["NUTS"]

###########################################
# !Overwrite pyro configuration behavior!
# currently this is the only way to provide
# mass matrix manually
###########################################
def new_configure(self, mass_matrix_shape, adapt_mass_matrix=True, options={}):
    """
    Sets up an initial mass matrix.

    :param dict mass_matrix_shape: a dict that maps tuples of site names to the shape of
        the corresponding mass matrix. Each tuple of site names corresponds to a block.
    :param bool adapt_mass_matrix: a flag to decide whether an adaptation scheme will be used.
    :param dict options: tensor options to construct the initial mass matrix.
    """
    inverse_mass_matrix = {}
    for site_names, shape in mass_matrix_shape.items():
        self._mass_matrix_size[site_names] = shape[0]
        diagonal = len(shape) == 1
        inverse_mass_matrix[site_names] = (
            torch.full(shape, self._init_scale, **options)
            if diagonal
            else torch.eye(*shape, **options) * self._init_scale
        )
        if adapt_mass_matrix:
            adapt_scheme = WelfordCovariance(diagonal=diagonal)
            self._adapt_scheme[site_names] = adapt_scheme

    if len(self.inverse_mass_matrix.keys()) == 0:
        self.inverse_mass_matrix = inverse_mass_matrix


BlockMassMatrix.configure = new_configure
############################################


class NUTS(BaseOptimizer):
    """No U-Turn Sampler (NUTS) implementation for Hamiltonian Monte Carlo
    (HMC) based MCMC sampling.

    This is a wrapper for the Pyro package: https://docs.pyro.ai/en/stable/index.html

    The NUTS class provides an implementation of the No-U-Turn Sampler
    (NUTS) algorithm, which is a variation of the Hamiltonian Monte
    Carlo (HMC) method for Markov Chain Monte Carlo (MCMC)
    sampling. This implementation uses the Pyro library to perform the
    sampling. The NUTS algorithm utilizes gradients of the target
    distribution to more efficiently explore the probability
    distribution of the model.

    More information on HMC and NUTS can be found at:
    https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo,
    https://arxiv.org/abs/1701.02434, and
    http://www.mcmchandbook.net/HandbookChapter5.pdf

    Args:
        model (AutoPhot_Model): The model which will be sampled.
        initial_state (Optional[Sequence], optional): A 1D array with the values for each parameter in the model. These values should be in the form of "as_representation" in the model. Defaults to None.
        max_iter (int, optional): The number of sampling steps to perform. Defaults to 1000.
        epsilon (float, optional): The step size for the NUTS sampler. Defaults to 1e-3.
        inv_mass (Optional[Tensor], optional): Inverse Mass matrix (covariance matrix) for the Hamiltonian system. Defaults to None.
        progress_bar (bool, optional): If True, display a progress bar during sampling. Defaults to True.
        prior (Optional[Distribution], optional): Prior distribution for the model parameters. Defaults to None.
        warmup (int, optional): Number of warmup (or burn-in) steps to perform before sampling. Defaults to 100.
        nuts_kwargs (Dict[str, Any], optional): A dictionary of additional keyword arguments to pass to the NUTS sampler. Defaults to {}.
        mcmc_kwargs (Dict[str, Any], optional): A dictionary of additional keyword arguments to pass to the MCMC function. Defaults to {}.

    Methods:
        fit(state: Optional[torch.Tensor] = None, nsamples: Optional[int] = None, restart_chain: bool = True) -> 'NUTS':
            Performs the MCMC sampling using a NUTS HMC and records the chain for later examination.

    """

    def __init__(
        self,
        model: "AutoPhot_Model",
        initial_state: Optional[Sequence] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.inv_mass = kwargs.get("inv_mass", None)
        self.epsilon = kwargs.get("epsilon", 1e-4)
        self.progress_bar = kwargs.get("progress_bar", True)
        self.prior = kwargs.get("prior", None)
        self.warmup = kwargs.get("warmup", 100)
        self.nuts_kwargs = kwargs.get("nuts_kwargs", {})
        self.mcmc_kwargs = kwargs.get("mcmc_kwargs", {})

    def fit(
        self,
        state: Optional[torch.Tensor] = None,
        nsamples: Optional[int] = None,
        restart_chain: bool = True,
    ):
        """
        Performs the MCMC sampling using a NUTS HMC and records the chain for later examination.
        """

        def step(model, prior):
            x = pyro.sample("x", prior)
            # Log-likelihood function
            log_likelihood_value = -model.negative_log_likelihood(
                parameters=x, as_representation=True
            )
            # Observe the log-likelihood
            pyro.factor("obs", log_likelihood_value)

        if self.prior is None:
            self.prior = dist.Normal(
                self.current_state,
                torch.ones_like(self.current_state) * 1e2
                + torch.abs(self.current_state) * 1e2,
            )

        # Set up the NUTS sampler
        nuts_kwargs = {
            "jit_compile": True,
            "ignore_jit_warnings": True,
            "step_size": self.epsilon,
            "full_mass": True,
            "adapt_step_size": True,
            "adapt_mass_matrix": self.inv_mass is None,
        }
        nuts_kwargs.update(self.nuts_kwargs)
        nuts_kernel = pyro_NUTS(step, **nuts_kwargs)
        if self.inv_mass is not None:
            nuts_kernel.mass_matrix_adapter.inverse_mass_matrix = {
                ("x",): self.inv_mass
            }

        # Provide an initial guess for the parameters
        init_params = {"x": self.model.parameters.get_vector(as_representation=True)}

        # Run MCMC with the NUTS sampler and the initial guess
        mcmc_kwargs = {
            "num_samples": self.max_iter,
            "warmup_steps": self.warmup,
            "initial_params": init_params,
            "disable_progbar": not self.progress_bar,
        }
        mcmc_kwargs.update(self.mcmc_kwargs)
        mcmc = pyro_MCMC(nuts_kernel, **mcmc_kwargs)

        mcmc.run(self.model, self.prior)
        self.iteration += self.max_iter

        # Extract posterior samples
        chain = mcmc.get_samples()["x"]

        with torch.no_grad():
            for i in range(len(chain)):
                chain[i] = self.model.parameters.transform(
                    chain[i], to_representation=False
                )
        self.chain = chain

        return self
