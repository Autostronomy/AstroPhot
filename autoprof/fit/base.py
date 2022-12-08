import torch
import numpy as np
from time import time

__all__ = ["BaseOptimizer"]

class BaseOptimizer(object):
    """
    Base optimizer object that other optimizers inherit from. Ensures consistent signature for the classes.

    Parameters:
        model: an AutoProf_Model object that will have it's (unlocked) parameters optimized [AutoProf_Model]
        initial_state: optional initialization for the parameters as a 1D tensor [tensor]
        max_iter: maximum allowed number of iterations [int]
        relative_tolerance: tolerance for counting success steps as: 0 < (Chi2^2 - Chi1^2)/Chi1^2 < tol [float]
    
    """
    def __init__(self, model, initial_state = None, max_iter = None, relative_tolerance = 1e-7, **kwargs):
        self.model = model
        self.verbose = kwargs.get("verbose", 0)
        
        if initial_state is None: 
            try:
                keys, reps = self.model.get_parameters_representation()
                allparams = []
                for R in reps:
                    assert not R is None
                    allparams += list(R.detach().cpu().numpy().flatten())
            except AssertionError:
                self.model.initialize()
                keys, reps = self.model.get_parameters_representation()
                allparams = []
                for R in reps:
                    allparams += list(R.detach().cpu().numpy().flatten())
            initial_state = torch.tensor(allparams, dtype = self.model.dtype, device = self.model.device)
        else:
            initial_state = torch.as_tensor(initial_state, dtype = self.model.dtype, device = self.model.device)
                
        self.current_state = torch.as_tensor(initial_state, dtype = self.model.dtype, device = self.model.device)
        if self.verbose > 1:
            print("initial state: ", self.current_state)
        if max_iter is None:
            self.max_iter = 100*len(initial_state)
        else:
            self.max_iter = max_iter
        self.iteration = 0

        self.relative_tolerance = relative_tolerance
        self.lambda_history = []
        self.loss_history = []
        self.message = ""

    def fit(self):
        pass
    def step(self, current_state = None):
        pass

    def res(self):
        N = np.isfinite(self.loss_history)
        return np.array(self.lambda_history)[N][np.argmin(np.array(self.loss_history)[N])]
