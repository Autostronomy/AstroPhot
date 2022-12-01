# Levenberg-Marquardt algorithm
import torch
import numpy as np
from time import time
from .base import BaseOptimizer

__all__ = ["LM"]

class LM(BaseOptimizer):
    """based heavily on:
    @article{gavin2019levenberg,
        title={The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems},
        author={Gavin, Henri P},
        journal={Department of Civil and Environmental Engineering, Duke University},
        volume={19},
        year={2019}
    }

    The Levenberg-Marquardt algorithm bridges the gap between a
    gradient descent optimizer and a Newton's Method optimizer. The
    Hessian for the Newton's Method update is too complex to evaluate
    with automatic differentiation (memory scales roughly as
    parameters^2 * pixels^2) and so an approximation is made using the
    Jacobian of the image pixels wrt to the parameters of the
    model. Automatic differentiation provides an exact Jacobian as
    opposed to a finite differences approximation.

    """
    
    def __init__(self, model, initial_state = None, **kwargs):
        super().__init__(model, initial_state, **kwargs)
        
        self.epsilon4 = kwargs.get("epsilon4", 1e-1)
        self.Lup = kwargs.get("Lup", 11.)
        self.Ldn = kwargs.get("Ldn", 9.)
        self.L = kwargs.get("L0", 1.)
        
        self.Y = self.model.target.data.view(-1)
        #        1 / sigma^2
        self.W = 1. / self.model.target.variance.view(-1) if model.target.has_variance else None
        #          # pixels      # parameters              # masked pixels
        self.ndf = len(self.Y) - len(self.current_state) - torch.sum(model.target.mask).item()
        self.J = None
        self.current_Y = None
        self.prev_Y = [None, None]

        self.L_history = []

    
    def step(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state
            
        print("---------iter---------")
        h = self.update_h()
                    
        with torch.no_grad():
            self.current_Y = self.model.full_sample(self.current_state + h).view(-1)
            loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if self.iteration > 0:
            print("LM loss: ", loss, np.min(self.loss_history[:-1]), h)
            rho = self.rho(np.min(self.loss_history[:-1]), loss, h) 
            print(rho)
            if rho > self.epsilon4:
                print("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / self.Ldn)
                self._count_reject = 0
                if 0 < (np.min(self.loss_history[:-1]) - loss) < 1e-6:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
            elif self._count_reject < 4:
                print("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1
                return
            else:
                print("reject")
                if self._count_reject == 4:
                    self.L = min(1e9, self.L / self.Lup**2)
                else:
                    self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1

        if self.J is None or self.iteration < 2 or self._count_reject >= 4:
            self.update_J_AD(h)
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)

        with torch.no_grad():
            self.update_hess()
            self.update_grad(self.current_Y)
        self.iteration += 1
    
    def fit(self):

        self.model.startup()
        self.model.step()
        
        self.iteration = 0
        self._count_reject = 0
        self._count_finish = 0

        try:
            while True:

                self.step()
            
                if self._count_finish >= 3:
                    print("success")
                    break
                elif self._count_reject >= 12:
                    print("fail reject 12 in a row")
                    break
                elif self.iteration >= self.max_iter:
                    print("fail max iterations reached: ", self.iteration)
                    break
                elif not torch.all(torch.isfinite(self.current_state)):
                    print("fail non-finite step taken")
                    break
        except KeyboardInterrupt:
            print("fail interrupted")
            
        self.model.finalize()

        return self
            
    def update_h(self):
            
        count_reject = 0
        h = torch.zeros(len(self.current_state))
        if self.iteration == 0:
            return h
        while count_reject < 4:
            # Sometimes the hesian + lambda matrix is singular, sometimes that can be fixed by giving lambda a boost.
            try:
                with torch.no_grad():
                    h = torch.linalg.solve(self.hess + self.L*torch.abs(torch.diag(self.hess))*torch.eye(len(self.grad)), self.grad)
                break
            except Exception as e:
                print("reject err: ", e)
                self.L = min(1e9, self.L * self.Lup)
                count_reject += 1
        return h
    
    def update_J_AD(self, h):
        self.J = self.model.jacobian(self.current_state + h).view(-1,len(self.current_state))
        
    def update_J_Broyden(self, h, Yp, Yph):
        with torch.no_grad():
            self.J += torch.outer(Yph - Yp - torch.matmul(self.J, h),h) / torch.linalg.norm(h)

    def update_hess(self):
        if self.W is None:
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W),-1)*self.J)
            
    def update_grad(self, Yph):
        if self.W is None:
            self.grad = torch.matmul(self.J.T, (self.Y - Yph))
        else:
            self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))
            
    def rho(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess)) * h) + self.grad))

    def res(self):
        return self.lambda_history[np.argmin(self.loss_history)]
