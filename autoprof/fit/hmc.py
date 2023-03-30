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

        self.epsilon = kwargs.get("epsilon", 1e-2)
        self.leapfrog_steps = kwargs.get("leapfrog_steps", 20)
        self.mass = torch.tensor(kwargs.get("mass", 1.0))
        self.temperature = torch.tensor(kwargs.get("temperature", 1.0))
        self.temper = torch.tensor(kwargs.get("temper", 1.0))

        self.Y = self.model.target[self.model.window].flatten("data")
        #        1 / sigma^2
        self.W = (
            1.0 / self.model.target[self.model.window].flatten("variance")
            if model.target.has_variance
            else 1.0
        )
        #          # pixels      # parameters
        self.ndf = len(self.Y) - len(self.current_state)

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
            self.chain = []
        else:
            self.chain = list(self.chain)
        for _ in tqdm(range(nsamples)):
            while (
                True
            ):  # rerun step function if it encounters a numerical error. Note that many such re-runs will bias the final posterior
                try:
                    state, score, chi2 = self.step(state, score, chi2)
                    break
                except RuntimeError:
                    warnings.warn(
                        "HMC numerical integration error, infinite momentum, consider smaller step size epsilon",
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
            loss = (
                torch.sum(((self.Y - Y) ** 2 * self.W)[torch.logical_not(self.mask)])
                / self.ndf
            )
        else:
            loss = torch.sum((self.Y - Y) ** 2 * self.W) / self.ndf

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
        momentum_0 = torch.normal(
            mean=torch.zeros_like(state), std=self.temperature * self.mass
        )
        momentum_t = torch.clone(momentum_0)
        x_t = torch.clone(state)
        score_t = torch.clone(score)
        temper = torch.sqrt(self.temper)
        for leap in range(self.leapfrog_steps):
            # Update step
            momentum_tp = (
                temper if leap < self.leapfrog_steps / 2 else (1 / temper)
            ) * (momentum_t + self.epsilon * score_t / 2)
            x_tp1 = x_t + self.epsilon * momentum_tp / self.mass
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
        DP = (
            0.5
            * (torch.dot(momentum_0, momentum_0) - torch.dot(momentum_t, momentum_t))
            / self.mass
        )
        log_alpha = (DU + DP) / self.temperature

        # Determine if proposal is accepted
        accept = self.accept(log_alpha)

        # Record result
        self._accepted += accept
        self._sampled += 1
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
