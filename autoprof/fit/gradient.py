# Traditional gradient descent with Adam
import torch
import numpy as np
from time import time
from .base import BaseOptimizer

__all__ = ["Grad"]

class Grad(BaseOptimizer):
    """Basic wrapper just to make it easier to use the built in pytorch
    gradient descent optimizers on AutoProf_Model objects. Note that
    most of the pytorch gradient descent optimizers are designed for
    stochastic gradient descent which is commonly used to optimize
    Neural Networks. In the case of astronomical image fitting the
    full Chi^2 can be computed at every iteration and so robustness to
    stochastic changes to the Chi^2 is not necessary in this
    case. Still, the methods like momentum which help for stochastic
    loss functions are also helpful for covariant parameters which are
    common in image fitting. When two models overlap considerably
    there is a covariance in their parameters, this can be challenging
    for a basic gradient descent optimizer but momentum helps with
    convergence. In general however, for most astronomical image
    fitting tasks it is faster to use a second order method such as
    the Levenberg-Marquardt algorithm (implimented in AutoProf as
    autoprof.fit.LM).

    The default method is "NAdam" which is a variant of the Adam
    algorithm. Adam performs gradient descent optimization with a
    momentum in the first and second moment of the gradient.
    Essentially momentum in the gradient and its square values. The
    NAdam method incorporate Nesterov momentum into the algorithm
    which sometimes has faster convergence properties.

    Parameters:
        model: and AutoProf_Model object with which to perform optimization [AutoProf_Model object]
        initial_state: optionally, and initial state for optimization [torch.Tensor]
        method: optimization method to use for the update step. Any optimizer in pytorch should work [str]
        patience: number of iterations without improvement before optimizer will exit early [int or None]
        optim_kwargs: dictionary of key word arguments to pass to the pytorch optimizer [dict]

    """

    
    def __init__(self, model, initial_state = None, **kwargs):

        super().__init__(model, initial_state, **kwargs)
        self.current_state.requires_grad = True

        # set parameters from the user
        self.patience = kwargs.get("patience", None)
        self.method = kwargs.get("method", "NAdam").strip()
        self.optim_kwargs = kwargs.get("optim_kwargs", {})

        # Default learning rate if none given. Equalt to 1 / sqrt(parames)
        if not "lr" in self.optim_kwargs:
            self.optim_kwargs["lr"] = 1. / np.sqrt(len(self.current_state))

        # Instantiates the appropriate pytorch optimizer with the initial state and user provided kwargs
        self.optimizer = getattr(torch.optim, self.method)((self.current_state,), **self.optim_kwargs)

    def step(self):
        """Take a single gradient step. Calls the model loss function,
        applies automatic differentiation to get the gradient of the
        parameters and takes a step with the pytorch optimizer.

        """
        
        self.iteration += 1
                
        self.optimizer.zero_grad()
        
        loss = self.model.full_loss(self.current_state, as_representation = True, override_locked = False)

        loss.backward()

        self.loss_history.append(loss.detach().cpu().item())
        self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
        if self.verbose > 0:
            print("loss: ", loss.item())
        if self.verbose > 1:
            print("gradient: ", self.current_state.grad)
        self.optimizer.step()
        
    def fit(self):
        """
        Perform an iterative fit of the model parameters using the specified optimizer
        """

        try:
            while True:
                self.step()
                if self.iteration >= self.max_iter:
                    self.message = self.message + " fail max iteration reached"
                    break
                if self.patience is not None and (len(self.loss_history) - np.argmin(self.loss_history)) > self.patience:
                    self.message = self.message + " fail no improvement"
                    break
                L = np.sort(self.loss_history)
                if len(L) >= 3 and 0 < L[1] - L[0] < 1e-6 and 0 < L[2] - L[1] < 1e-6:
                    self.message = self.message + " success"
                    break
        except KeyboardInterrupt:
            self.message = self.message + " fail interrupted"

        # Set the model parameters to the best values from the fit and clear any previous model sampling
        self.model.set_parameters(torch.tensor(self.res()), as_representation = True, override_locked = False)
        # finalize tells the model that optimization is now finished
        self.model.finalize()
        
        return self
        
