# Hamiltonian Monte-Carlo
import os
from time import time
from typing import Optional, Sequence
import warnings
import torch
from tqdm import tqdm
import numpy as np
from .base import BaseOptimizer
from .. import AP_config

__all__ = ["HMC"]


class HMC(BaseOptimizer):
    """Hamiltonian Monte-Carlo sampler, based on:
    https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo and
    https://arxiv.org/abs/1701.02434 and
    http://www.mcmchandbook.net/HandbookChapter5.pdf. This MCMC
    algorithm uses gradients of the Chi^2 to more efficiently explore
    the probability distribution.

    Args:
      model (AutoProf_Model): The model which will be sampled.
      initial_state (Optional[Sequence]): A 1D array with the values for each parameter in the model. Note that these values should be in the form of "as_representation" in the model.
      max_iter (int): The number of sampling steps to perform. Default 1000
      epsilon (float): The length of the integration step to perform for each leapfrog iteration. The momentum update will be of order elipson * score. Default 1e-2
      leapfrog_steps (int): Number of steps to perform with leapfrog integrator per sample of the HMC. Default 20
      mass (float or array): Mass vector which can tune the behavior in each dimension to ensure better mixing when sampling. Default 1.

    """

    def __init__(
        self,
        model: "AutoProf_Model",
        initial_state: Optional[Sequence] = None,
        max_iter: int = 1000,
        **kwargs
    ):
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.epsilon = kwargs.get("epsilon", 1e-5)
        self.leapfrog_steps = kwargs.get("leapfrog_steps", 20)
        self.mass = kwargs.get("mass", None)
        self.temperature = torch.tensor(kwargs.get("temperature", 1.0), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        self.temper = torch.tensor(kwargs.get("temper", 1.0), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        self.progress_bar = kwargs.get("progress_bar", True)
        self.min_accept = kwargs.get("min_accept", 0.1)

        self.Y = self.model.target[self.model.window].flatten("data")
        #        1 / sigma^2
        self.W = (
            1.0 / self.model.target[self.model.window].flatten("variance")
            if model.target.has_variance
            else 1.0
        )

        self.reset_chain()

    def reset_chain(self):
        self.chain = []
        self._accepted = 0
        self._sampled = 0
        
    def fit(
        self,
        state: Optional[torch.Tensor] = None,
        nsamples: Optional[int] = None,
        restart_chain: bool = True,
    ):
        """
        Performs the MCMC sampling using a Hamiltonian Monte-Carlo step and records the chain for later examination.
        """

        if nsamples is None:
            nsamples = self.max_iter

        if state is None:
            state = self.current_state
        score, chi2 = self.score_fn(state)

        if restart_chain:
            self.reset_chain()
        else:
            self.chain = list(self.chain)
        for _ in self.iter_generator(nsamples):
            while (
                True
            ):  # rerun step function if it encounters a numerical error. Note that many such re-runs will bias the final posterior
                try:
                    state, score, chi2 = self.step(state, score, chi2)
                    break
                except RuntimeError as e:
                    print("Error encountered. Reducing step size epsilon by factor 10")
                    self.epsilon /= 10.
                    warnings.warn(
                        "HMC numerical integration error. Perhaps rerun with smaller step size.",
                        RuntimeWarning,
                    )

            self.append_chain(state)
        self.current_state = state
        self.chain = np.stack(self.chain)
        return self

    def append_chain(self, state: torch.Tensor) -> None:
        """
        Add a state vector to the MCMC chain
        """
        self.model.set_parameters(state, as_representation=True)
        chain_state = self.model.get_parameter_vector(as_representation=False)
        self.chain.append(chain_state.detach().cpu().clone().numpy())

    def score_fn(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Compute the score for the current state. This is the gradient of the Chi^2 wrt the model parameters.
        """
        # Set up state with grad
        gradstate = torch.as_tensor(state).clone().detach()
        gradstate.requires_grad = True

        # Sample model
        Y = self.model(parameters=gradstate, as_representation=True).flatten("data")

        # Compute Chi^2
        if self.model.target.has_mask:
            loss = torch.sum(
                ((self.Y - Y) ** 2 * self.W)[torch.logical_not(self.mask)]
            ) / 2.
        else:
            loss = torch.sum(
                (self.Y - Y) ** 2 * self.W
            ) / 2.

        # Compute score
        loss.backward()

        return -gradstate.grad, loss.detach()

    @staticmethod
    def accept(log_alpha) -> torch.Tensor:
        """
        Evaluates randomly if a given proposal is accepted. This is done in log space which is more natural for the evaluation in the step.
        """
        return torch.log(torch.rand(log_alpha.shape)) < log_alpha

    def step(
        self, state: torch.Tensor, score: torch.Tensor, chi2: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes one step of the HMC sampler by integrating along a path initiated with a random momentum.
        """
        momentum_0 = torch.distributions.MultivariateNormal(
            loc = torch.zeros_like(state),
            covariance_matrix = self.mass
        ).sample()
        momentum_t = torch.clone(momentum_0)
        x_t = torch.clone(state)
        score_t = torch.clone(score)
        temper = torch.sqrt(self.temper)
        for leap in range(self.leapfrog_steps):
            # Update step
            momentum_tp = (
                temper if leap < self.leapfrog_steps / 2 else (1 / temper)
            ) * (momentum_t + self.epsilon * score_t / 2)
            x_tp1 = x_t + self.epsilon * (self._inv_mass @ momentum_tp)
            score_tp1, chi2_tp1 = self.score_fn(x_tp1)
            momentum_tp1 = (
                temper if leap < self.leapfrog_steps // 2 else (1 / temper)
            ) * (momentum_tp + self.epsilon * score_tp1 / 2)

            # set for next step
            x_t = torch.clone(x_tp1)
            momentum_t = momentum_tp1
            score_t = torch.clone(score_tp1)

            # Test for failure case
            if torch.any(torch.logical_not(torch.isfinite(momentum_t))):
                raise RuntimeError(
                    "HMC numerical integration error, infinite momentum, consider smaller step size epsilon"
                )
            

        # Set the proposed values as the end of the leapfrog integration
        proposal_state = x_t
        proposal_chi2 = chi2_tp1
        proposal_score = score_tp1

        # Evaluate the Hamiltonian likelihood
        DU = chi2 - proposal_chi2
        DP = 0.5 * (
            (momentum_0 @ self._inv_mass @ momentum_0) - (momentum_t @ self._inv_mass @ momentum_t)
        )
        log_alpha = (DU + DP) / self.temperature

        # Determine if proposal is accepted
        accept = self.accept(log_alpha)

        # Record result
        self._accepted += accept
        self._sampled += 1

        if len(self.chain) > 100 and self.acceptance() < self.min_accept:
            raise RuntimeError("HMC acceptance too low, consider smaller step size.")
        
        return (
            (proposal_state, proposal_score, proposal_chi2)
            if accept
            else (state, score, chi2)
        )

    @property
    def acceptance(self):
        """
        Returns the ratio of accepted states to total states sampled.
        """
        return self._accepted / self._sampled

    @property
    def mass(self):
        return self._mass
    @mass.setter
    def mass(self, value):
        """Set the mass matrix for the HMC sampler

        A note when setting the mass matrix it is often a good idea to
        set it to `mass / mean(mass)` to normalize the matrix.
        Otherise it is possible for the numerical stability to be off
        if there is a huge discrepancy between the parameters and the
        momentum. This can show up as requring a very small epsilon
        for the chain to run, which then leaves a high
        autocorrelation.

        """
        if value is None:
            value = torch.eye(
                len(self.current_state),
                dtype = AP_config.ap_dtype,
                device = AP_config.ap_device
            )
        self._mass = torch.as_tensor(value, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        self._inv_mass = torch.linalg.inv(self._mass)
        self._det_mass = torch.linalg.det(self._mass)

    def iter_generator(self, N):
        if self.progress_bar:
            return tqdm(range(N))
        return range(N)
        
    def estimate_mass(self, chain = None):
        if chain is None:
            chain = self.chain

        return np.cov(chain, rowvar = False)
