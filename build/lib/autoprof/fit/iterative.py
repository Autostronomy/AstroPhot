# Apply a different optimizer iteratively
import os
import torch
import numpy as np
from time import time
from .base import BaseOptimizer
import matplotlib.pyplot as plt

__all__ = ["Iter"]

class Iter(BaseOptimizer):
    """This is an optimizer wrapper which performs optimization by
    iteratively applying a different optimizer to a group model. This
    can sometimes be advantageous for fitting an extremely large
    number of models, more than can fit in memory, or for complex fits
    in which the degeneracies of parameters may overwhelm a fitting
    which acts simultaneously on all models.

    Parameters:
        model: and AutoProf_Model object with which to perform optimization [AutoProf_Model object]
        method: optimizer class to apply at each iteration step [BaseOptimizer object]
        initial_state: optionally, and initial state for optimization [torch.Tensor]
        
    """

    def __init__(self, model, method, initial_state = None, max_iter = 100, method_kwargs = {}, **kwargs):

        super().__init__(model, initial_state, max_iter = max_iter, **kwargs)

        self.method = method
        self.method_kwargs = method_kwargs
        #          # pixels      # parameters
        self.ndf = self.model.target[self.model.window].flatten("data").size(0) - len(self.current_state)
        if self.model.target.has_mask:
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(self.model.target[self.model.window].flatten("mask")).item()
        
    def sub_step(self, model):
        self.Y -= model.sample()
        model.target = model.target[model.window] - self.Y[model.window]
        res = self.method(model, **self.method_kwargs).fit()
        self.Y += model.sample()
        if self.verbose > 1:
            print(res.message)
        model.target = self.model.target
        
    def step(self):
        if self.verbose > 0:
            print("--------iter-------")

        # Fit each model individually
        for model in self.model.model_list:
            if self.verbose > 0:
                print(model.name)
            self.sub_step(model)
        # Update the current state
        self.current_state = self.model.get_parameter_vector(as_representation = True)

        # update the loss value
        with torch.no_grad():
            self.Y = self.model.full_sample(self.current_state, as_representation = True, override_locked = False, return_data = False)
            D = self.model.target[self.model.window].flatten("data")
            V = self.model.target[self.model.window].flatten("variance") if self.model.target.has_variance else 1.
            if self.model.target.has_mask:
                M = self.model.target[self.model.window].flatten("mask")
                loss = torch.sum((((D - self.Y.flatten("data"))**2 ) / V)[torch.logical_not(M)]) / self.ndf
            else:
                loss = torch.sum(((D - self.Y.flatten("data"))**2 / V)) / self.ndf
        if self.verbose > 0:
            print("Loss: ", loss.item())
        self.lambda_history.append(np.copy((self.current_state).detach().cpu().numpy()))
        self.loss_history.append(loss.item())
        
        # test for convergence
        if self.iteration > 2 and (0 < ((self.loss_history[-2] - self.loss_history[-1])/self.loss_history[-1]) < self.relative_tolerance):
            self._count_finish += 1
        else:
            self._count_finish = 0

        self.iteration += 1
        
    def fit(self):

        self.iteration = 0
        self.Y = self.model.full_sample(self.current_state, as_representation = True, override_locked = False, return_data = False)

        try:
            while True:
                self.step()
                if self.save_steps is not None:
                    self.model.save(os.path.join(self.save_steps, f"{self.model.name}_Iteration_{self.iteration:03d}.yaml"))
                if self.iteration > 2 and self._count_finish > 3:
                    self.message = self.message + "success"
                    break                    
                elif self.iteration > self.max_iter:
                    self.message = self.message + f"fail max iterations reached: {self.iteration}"
                    break
                    

        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"
            
        self.model.set_parameters(torch.tensor(self.res(), dtype = self.model.dtype, device = self.model.device), as_representation = True, override_locked = False)
        self.model.finalize()
            
        return self
