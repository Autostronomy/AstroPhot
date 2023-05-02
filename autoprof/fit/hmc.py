# Hamiltonian Monte-Carlo
import os
from time import time
from typing import Optional, Sequence
import warnings

import torch
from tqdm import tqdm
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC as pyro_MCMC
from pyro.infer import HMC as pyro_HMC

from .base import BaseOptimizer
from .. import AP_config

__all__ = ["HMC"]

class HMC(BaseOptimizer):
    """Hamiltonian Monte-Carlo sampler wrapper for the Pyro package.

    This MCMC algorithm uses gradients of the Chi^2 to more
    efficiently explore the probability distribution. Consider using
    the NUTS sampler instead of HMC, as it is generally better in most
    aspects.

    More information on HMC can be found at:
    https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo,
    https://arxiv.org/abs/1701.02434, and
    http://www.mcmchandbook.net/HandbookChapter5.pdf

    Args:
        model (AutoProf_Model): The model which will be sampled.
        initial_state (Optional[Sequence]): A 1D array with the values for each
            parameter in the model. Note that these values should be in the form
            of "as_representation" in the model.
        max_iter (int, optional): The number of sampling steps to perform.
            Default is 1000.
        epsilon (float, optional): The length of the integration step to perform
            for each leapfrog iteration. The momentum update will be of order
            elipson * score. Default is 1e-5.
        leapfrog_steps (int, optional): Number of steps to perform with leapfrog
            integrator per sample of the HMC. Default is 20.
        mass_matrix (float or array, optional): Mass matrix which can tune the
            behavior in each dimension to ensure better mixing when sampling.
            Default is the identity.

    """
    
    def __init__(
        self,
        model: "AutoProf_Model",
        initial_state: Optional[Sequence] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.epsilon = kwargs.get("epsilon", 1e-3)
        self.leapfrog_steps = kwargs.get("leapfrog_steps", 20)
        self.progress_bar = kwargs.get("progress_bar", True)
        self.prior = kwargs.get("prior", None)
        self.warmup = kwargs.get("warmup", 100)
        self.hmc_kwargs = kwargs.get("hmc_kwargs", {})
        self.mcmc_kwargs = kwargs.get("mcmc_kwargs", {})
        self.acceptance = None
        
        if "mass_matrix" not in self.hmc_kwargs and "mass_matrix" in kwargs:
            self.hmc_kwargs["mass_matrix"] = kwargs.get("mass_matrix")

    def fit(
        self,
        state: Optional[torch.Tensor] = None,
    ):
        """Performs MCMC sampling using Hamiltonian Monte-Carlo step.

        Records the chain for later examination.

        Args:
            state (torch.Tensor, optional): Model parameters as a 1D tensor.

        Returns:
            HMC: An instance of the HMC class with updated chain.

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

        # Set up the HMC sampler
        hmc_kwargs = {
            "jit_compile": True,
            "ignore_jit_warnings": True,
            "full_mass": True,
            "step_size": self.epsilon,
            "num_steps": self.leapfrog_steps,
            "adapt_step_size": False,
        }
        hmc_kwargs.update(self.hmc_kwargs)
        hmc_kernel = pyro_HMC(step, **hmc_kwargs)

        # Provide an initial guess for the parameters
        init_params = {"x": self.model.get_parameter_vector(as_representation=True)}

        # Run MCMC with the HMC sampler and the initial guess
        mcmc_kwargs = {
            "num_samples": self.max_iter,
            "warmup_steps": self.warmup,
            "initial_params": init_params,
            "disable_progbar": not self.progress_bar,
        }
        mcmc_kwargs.update(self.mcmc_kwargs)
        mcmc = pyro_MCMC(hmc_kernel, **mcmc_kwargs)
        
        mcmc.run(self.model, self.prior)
        self.iteration += self.max_iter

        # Extract posterior samples
        chain = mcmc.get_samples()["x"]

        with torch.no_grad():
            for i in range(len(chain)):
                chain[i] = self.model.transform(chain[i], to_representation = False)
        self.chain = chain
        
        return self
