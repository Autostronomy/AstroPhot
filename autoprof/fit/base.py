import torch
import numpy as np
from time import time
from scipy.special import gammainc
from scipy.optimize import minimize
from .. import AP_config

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
    def __init__(self, model, initial_state = None, relative_tolerance = 1e-3, **kwargs):
        self.model = model
        self.verbose = kwargs.get("verbose", 0)
        
        if initial_state is None: 
            try:
                initial_state = self.model.get_parameter_vector(as_representation = True)
            except AssertionError:
                self.model.initialize()
                initial_state = self.model.get_parameter_vector(as_representation = True)
        else:
            initial_state = torch.as_tensor(initial_state, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
                
        self.current_state = torch.as_tensor(initial_state, dtype = AP_config.ap_dtype, device = AP_config.ap_device)
        if self.verbose > 1:
            print("initial state: ", self.current_state)
        self.max_iter = kwargs.get("max_iter", 100*len(initial_state))
        self.iteration = 0
        self.save_steps = kwargs.get("save_steps", None)
        
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

    def chi2contour(self, n_params, confidence = 0.682689492137):
            
        def _f(x, nu):
            return (gammainc(nu/2, x/2) - confidence)**2

        for method in ["L-BFGS-B", "Powell", "Nelder-Mead"]:
            res = minimize(_f, x0 = n_params, args = (n_params,), method = method, tol = 1e-8)

            if res.success:
                return res.x[0]
        raise RuntimeError(f"Unable to compute Chi^2 contour for ndf: {ndf}")
