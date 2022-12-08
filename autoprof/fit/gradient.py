# Traditional gradient descent with Adam
import torch
import numpy as np
from time import time
from .base import BaseOptimizer

__all__ = ["Grad"]

class Grad(BaseOptimizer):

    def __init__(self, model, initial_state = None, **kwargs):

        super().__init__(model, initial_state, **kwargs)
        self.current_state.requires_grad = True

        self.patience = kwargs.get("patience", 25)
        self.method = kwargs.get("method", "NAdam").strip()
        self.optim_kwargs = kwargs.get("optim_kwargs", {})
        
        if self.method == "NAdam":
            if not "lr" in self.optim_kwargs:
                self.optim_kwargs["lr"] = 1. / np.sqrt(len(self.current_state))
            self.optimizer = torch.optim.NAdam(
                (self.current_state,), **self.optim_kwargs
            )
        elif self.method == "Adam":
            if not "lr" in self.optim_kwargs:
                self.optim_kwargs["lr"] = 1. / np.sqrt(len(self.current_state))
            self.optimizer = torch.optim.Adam(
                (self.current_state,), **self.optim_kwargs
            )
        elif self.method == "LBFGS":
            if not "lr" in self.optim_kwargs:
                self.optim_kwargs["lr"] = 1. / np.sqrt(len(self.current_state))
            self.optimizer = torch.optim.LBFGS(
                (self.current_state,), **self.optim_kwargs
            )

    def step(self):
        
        self.iteration += 1
                
        self.optimizer.zero_grad()
        
        loss = self.model.full_loss(self.current_state)

        loss.backward()

        self.loss_history.append(loss.detach().cpu().item())
        self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
        if self.verbose > 0:
            print("loss: ", loss.item())
        if self.verbose > 1:
            print("gradient: ", self.current_state.grad)
        self.optimizer.step()
        
    def fit(self):
        
        self.model.startup()
        self.model.step()

        try:
            while True:
                self.step()
                if self.iteration >= self.max_iter:
                    self.message = self.message + " fail max iteration reached"
                    break
                if (len(self.loss_history) - np.argmin(self.loss_history)) > self.patience:
                    self.message = self.message + " fail no improvement"
                    break
                L = np.sort(self.loss_history)
                if len(L) >= 3 and 0 < L[1] - L[0] < 1e-6 and 0 < L[2] - L[1] < 1e-6:
                    self.message = self.message + " success"
                    break
        except KeyboardInterrupt:
            self.message = self.message + " fail interrupted"
            
        self.model.step(torch.tensor(self.res()))
        self.model.finalize()
        
        return self
        
