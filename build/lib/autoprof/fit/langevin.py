# Langevin Dynamics
import torch
import numpy as np
from time import time
from .base import BaseOptimizer

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
        print(x[i])
        y[i] = f(x[i])
    S = dx / 3 * torch.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S

class MALA(BaseOptimizer):
    """
    Metropolis Adjusted Langevin Algorithm.

    Borrowed heavily from Alexandre Adam: https://github.com/AlexandreAdam/
    """
    def __init__(self, model, initial_state = None, **kwargs):
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
                    print("max iter reached")
                    break
        except KeyboardInterrupt:
            print("interrupted")
            
        self.model.finalize()
            
        return self
            
    def step(self, current_state):
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
    
    def accept(self, log_alpha):
        """
        log_alpha: The log of the acceptance ratio. In details, this
            is the difference between the log probability of the proposed state x'
            and the previous state x, plus the difference between
            the transition kernel probability q in case it is asymmetrical:
                delta_alpha = log p(x') - log p(x) + log q(x | x') - log q(x' | x)
        :return: A boolean vector, whether the state is accepted or not
        """
        return torch.log(torch.rand(log_alpha.shape, device=self.model.device)) < log_alpha

    def delta_logp(self, current_state, proposed_state):
        """
        Computes the log probability difference between the proposed state x' and the current state x
            log p(x') - log p(x)
        """
        v = proposed_state - current_state
        T = self.delta_logp_steps + 1
        D = current_state.shape

        def integrand(t):
            """
            No assumption is made about the dimension of the state vector, other than
            it can be decomposed into a batch (or walkers) dimension and a state dimension (shape = [B, *D]).
            The dot product assumes a uniform weight matrix.
            t: Time vector of size T
            :return: The integrand, which is a vector of size [T, B] with value score(gamma(t))^T gamma'(t)
                with gamma(t) = t * v + cs
            """
            gamma_t = t * v + current_state  # broadcast onto new time dimension
            score = self.score_fn(gamma_t)  # compress time and batch dimension together for network
            #v_repeated = v.repeat(T, *[1]*len(D))  # copy batch T times to match compressed dimension
            return torch.dot(score, v) #torch.einsum("td, td -> t", score.view(T, -1), v_repeated.view(T, -1)).view(T) # compute dot product across D
        return simps(integrand, 0., 1., self.delta_logp_steps, device=self.model.device)

    def score_fn(self, state):
        state = torch.clone(state)
        state.requires_grad = True
        score = -2*self.model.full_loss(state)
        score.backward()
        return state.grad
    
    @property
    def acceptance(self):
        return self._accepted / self._sampled
