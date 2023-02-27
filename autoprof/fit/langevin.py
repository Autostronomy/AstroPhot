# Langevin Dynamics
import torch
import numpy as np
from time import time
from typing import Union, Dict, Optional
from .base import BaseOptimizer
from .. import AP_config

__all__ = ["MALA"]


def simps(f, a:float, b:float, T:int=128, device=None) -> torch.Tensor:
    """
    Approximate the integral of f(x) from a to b by Simpson's 1/3 rule.
    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.
    Parameters
    ----------
    f : function
        Torch function taking in a vector and returning a Tensor with inner dimension T
    a , b : numbers
        Interval of integration [a,b]
    T : (even) integer
        Number of subintervals of [a,b]
    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.
    Examples
    --------
    # >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    if T % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / T
    x = torch.linspace(a, b, T + 1, device=device)
    y = torch.zeros_like(x)
    for i in range(len(x)):
        y[i] = f(x[i])
    S = dx / 3 * torch.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

class MALA(BaseOptimizer):
    """
    Metropolis Adjusted Langevin Algorithm.

    Borrowed heavily from Alexandre Adam: https://github.com/AlexandreAdam/
    """
    def __init__(self, model: 'AutoProf', initial_state: Optional[Dict[str,torch.Tensor]] = None, **kwargs)-> None:
        """
    Initialize a Langevin Dynamics sampler object for AutoProf.

    Args:
        model (AutoProf): An AutoProf object representing the model to be sampled.
        initial_state (dict, optional): A dictionary containing the initial values of the model parameters to be sampled. 
                                        If None, the initial state is set to the current state of the model. 
                                        Defaults to None.
        **kwargs: Additional keyword arguments.
            epsilon (float): The step size of the Langevin Dynamics sampler. Defaults to 0.02.
            delta_logp_steps (int): The number of steps between the gradient evaluations used for 
                                    estimating the change in the log probability. 
                                    Defaults to 10.
            score_fn (callable): A function to evaluate the score (gradient of the log probability) of the model. 
                                If None, the score is computed using automatic differentiation in PyTorch. 
                                Defaults to None.
        """
        super().__init__(model, initial_state, **kwargs)

        self.epsilon = kwargs.get("epsilon", 0.02)
        self.delta_logp_steps = int(kwargs.get("delta_logp_steps", 10))
        
        if "score_fn" in kwargs:
            self.score_fn = kwargs["score_fn"]

    def fit(self):

        self.iteration = 0
        self._accepted = 0
        self._sampled = 0

        try:
            while True:
                
                self.current_state = self.step(self.current_state)
                self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
                
                if self.iteration >= self.max_iter:
                    self.message = self.message + "fail. max iterations reached"
                    break
        except KeyboardInterrupt:
            self.message = self.message + "fail. interrupted"
                        
        return self
            
    def step(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Take a step of Langevin Monte Carlo and return the new state.

        Args:
            current_state: The current state of the algorithm.

        Returns:
            A tensor representing the new state of the algorithm.
        """
        current_score = self.score_fn(current_state)
        proposed_state = current_state + self.epsilon * current_score + np.sqrt(2 * self.epsilon) * torch.normal(mean=torch.zeros_like(current_state), std=1)
        proposed_score = self.score_fn(proposed_state)
        delta_logp = self.delta_logp(current_state, proposed_state)
        kernel_forward = torch.sum((proposed_state - current_state - self.epsilon * current_score) ** 2) / 4 / self.epsilon
        kernel_backward = torch.sum((current_state - proposed_state - self.epsilon * proposed_score) ** 2) / 4 / self.epsilon
        log_alpha = delta_logp - kernel_backward + kernel_forward
        accepted = self.accept(log_alpha)
        if accepted:
            current_state = proposed_state
        self._accepted += accepted
        self._sampled += 1
        self.iteration += 1
        return current_state
    
    def accept(self, log_alpha: torch.Tensor)-> Union[bool, torch.Tensor]:
        """
        log_alpha: The log of the acceptance ratio. In details, this
            is the difference between the log probability of the proposed state x'
            and the previous state x, plus the difference between
            the transition kernel probability q in case it is asymmetrical:
                delta_alpha = log p(x') - log p(x) + log q(x | x') - log q(x' | x)
        :return: A boolean vector, whether the state is accepted or not
        """
        return torch.log(torch.rand(log_alpha.shape, device=self.model.device)) < log_alpha

    def delta_logp(self, current_state: torch.Tensor, proposed_state: torch.Tensor)-> torch.Tensor:
        """
        Computes the log probability difference between the proposed state x' and the current state x
            log p(x') - log p(x)
        """
        v = proposed_state - current_state
        T = self.delta_logp_steps + 1
        D = current_state.shape
        return simps(self.integrand, 0., 1., self.delta_logp_steps, device=self.model.device)

    def integrand(self, t: torch.Tensor)-> torch.Tensor:
            """
            No assumption is made about the dimension of the state vector, other than
            it can be decomposed into a batch (or walkers) dimension and a state dimension (shape = [B, *D]).
            The dot product assumes a uniform weight matrix.
            t: Time vector of size T
            
            returns : 
            The integrand, which is a vector of size [T, B] with value score(gamma(t))^T gamma'(t)
            with gamma(t) = t * v + cs
            """
            gamma_t = t * v + current_state  # broadcast onto new time dimension
            score = self.score_fn(gamma_t)  # compress time and batch dimension together for network
            #v_repeated = v.repeat(T, *[1]*len(D))  # copy batch T times to match compressed dimension
            return torch.dot(score, v) #torch.einsum("td, td -> t", score.view(T, -1), v_repeated.view(T, -1)).view(T) # compute dot product across D
        

    def score_fn(self, state: torch.Tensor)-> torch.Tensor:
        """
        Computes the score of a given state using the negative loss of the AutoProf model. 
        Uses automatic differentiation to compute the gradient of the state.

        Parameters:
            state: A tensor representing the current state.

        Returns:
            A tensor representing the score of the current state.
        """
        state = torch.clone(state)
        state.requires_grad = True
        score = -2*self.model.full_loss(state)
        score.backward()
        return state.grad
    
    @property
    def acceptance(self)-> float:
        """
        Calculates the acceptance ratio of the samples generated by the sampler.

        Returns:
            A float representing the acceptance ratio of the samples generated by the sampler.
        """
        return self._accepted / self._sampled
