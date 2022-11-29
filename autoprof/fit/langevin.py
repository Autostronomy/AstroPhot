# Langevin Dynamics
import torch
import numpy as np
from time import time

__all__ = ["MALA"]

class MALA(object):
    """
    Metropolis Adjusted Langevin Algorithm.

    Borrowed heavily from Alexandre Adam: https://github.com/AlexandreAdam/
    """
    def __init__(self, model, lambda0 = None, max_iter = None, run_fit = True):

        self.max_iter = 1000*len(lambda0)

        if run_fit:
            self.main_loop()
            
    def step(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state
        
        proposed_state = current_state + self.epsilon * self.score_fn(current_state) + np.sqrt(2 * self.epsilon) * torch.normal(mean=torch.zeros_like(current_state), std=1)
        delta_logp = self.delta_logp(current_state, proposed_state)
        kernel_forward = torch.sum((proposed_state - current_state - self.epsilon * self.score_fn(current_state)) ** 2, dim=dims) / 4 / self.epsilon
        kernel_backward = torch.sum((current_state - proposed_state - self.epsilon * self.score_fn(proposed_state)) ** 2, dim=dims) / 4 / self.epsilon
        log_alpha = delta_logp - kernel_backward + kernel_forward
        accepted = self.accept(log_alpha)
        if accepted:
            self.current_state = proposed_state
        self._accepted += accepted
        self._sampled += 1
        return self.current_state
    
    def accept(self, log_alpha):
        """
        log_alpha: The log of the acceptance ratio. In details, this
            is the difference between the log probability of the proposed state x'
            and the previous state x, plus the difference between
            the transition kernel probability q in case it is asymmetrical:
                delta_alpha = log p(x') - log p(x) + log q(x | x') - log q(x' | x)
        :return: A boolean vector, whether the state is accepted or not
        """
        return torch.log(torch.rand(log_alpha.shape, device=self.device)) < log_alpha

    def delta_logp(self, current_state, proposed_state):
        """
        Computes the log probability difference between the proposed state x' and the current state x
            log p(x') - log p(x)
        """
        v = proposed_state - current_state
        cs = current_state
        T = self.delta_logp_steps + 1
        B, *D = cs.shape

        def integrand(t):
            """
            No assumption is made about the dimension of the state vector, other than
            it can be decomposed into a batch (or walkers) dimension and a state dimension (shape = [B, *D]).
            The dot product assumes a uniform weight matrix.
            t: Time vector of size T
            :return: The integrand, which is a vector of size [T, B] with value score(gamma(t))^T gamma'(t)
                with gamma(t) = t * v + cs
            """
            gamma_t = t.view(T, 1, *[1]*len(D)) * v.unsqueeze(0) + cs.unsqueeze(0)  # broadcast onto new time dimension
            score = self.score_fn(gamma_t.view(T * B, *D))  # compress time and batch dimension together for network
            v_repeated = v.repeat(T, *[1]*len(D))  # copy batch T times to match compressed dimension
            return torch.einsum("td, td -> t", score.view(T*B, -1), v_repeated.view(T*B, -1)).view(T, B) # compute dot product across D
        return simps(integrand, 0., 1., self.delta_logp_steps, device=self.device)

    @property
    def acceptance(self):
        return self._accepted / self._sampled
