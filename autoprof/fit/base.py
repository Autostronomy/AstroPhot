import torch
import numpy as np
from time import time

__all__ = ["BaseOptimizer"]

class BaseOptimizer(object):
    """
    Base optimizer object that other optimizers inherit from. Ensures consistent signature for the classes.
    
    """
    def __init__(self, model, initial_state = None, max_iter = None, **kwargs):
        self.model = model
        self.verbose = kwargs.get("verbose", 0)
        
        if initial_state is None: # fixme, try to request parameters first
            self.model.initialize()
            keys, reps = self.model.get_parameters_representation()
            allparams = []
            for R in reps:
                allparams += list(R.detach().cpu().numpy().flatten())
            initial_state = torch.tensor(allparams, dtype = self.model.dtype, device = self.model.device)
                
        self.current_state = torch.as_tensor(initial_state, dtype = self.model.dtype, device = self.model.device)
        if self.verbose > 1:
            print("initial state: ", self.current_state)
        if max_iter is None:
            self.max_iter = 100*len(initial_state)
        else:
            self.max_iter = max_iter
        self.iteration = 0
        
        self.lambda_history = []
        self.loss_history = []
        self.message = ""

    def fit(self):
        pass
    def step(self, current_state = None):
        pass
