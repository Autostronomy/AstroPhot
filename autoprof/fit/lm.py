# Levenberg-Marquardt algorithm
import torch
import numpy as np
from time import time
from .base import BaseOptimizer
from .gradient import Grad
import matplotlib.pyplot as plt

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
        
        self.epsilon4 = kwargs.get("epsilon4", 0)
        self.Lup = kwargs.get("Lup", 11.)
        self.Ldn = kwargs.get("Ldn", 9.)
        self.L = kwargs.get("L0", 1.)
        
        self.Y = self.model.target[self.model.fit_window].data.reshape(-1)
        #        1 / sigma^2
        self.W = 1. / self.model.target[self.model.fit_window].variance.reshape(-1) if model.target.has_variance else None
        #          # pixels      # parameters              # masked pixels
        self.ndf = len(self.Y) - len(self.current_state) - torch.sum(model.target[self.model.fit_window].mask).item()
        self.J = None
        self.current_Y = None
        self.prev_Y = [None, None]
        if self.model.target.has_mask:
            self.mask = self.model.target[self.model.fit_window].mask.reshape(-1)
        self.L_history = []

    def grad_step(self):

        print(self.current_state)
        print(self.model.full_loss(self.current_state))
        grad_res = Grad(self.model, self.current_state, max_iter = 20, optim_kwargs = {"lr": 1e-3}).fit()
        self.current_state = torch.tensor(grad_res.lambda_history[np.argmin(grad_res.loss_history)], dtype = self.model.dtype, device = self.model.device)
        print(self.current_state)
        print(self.model.full_loss(self.current_state)) 
        plt.plot(range(len(grad_res.loss_history)), grad_res.loss_history)
        plt.show()
        # temp_state = torch.clone(self.current_state)
        # temp_state.requires_grad = True
        # loss = self.model.full_loss(temp_state)
        # loss.backward()
        # print(loss)
        # self.grad = -temp_state.grad
        # self.hess = torch.eye(len(self.grad))
        # next_loss = 2*loss
        # self.L = 1e1
        # while next_loss > loss:
        #     print("pure grad loop")
        #     self.L *= 5.
        #     with torch.no_grad():
        #         next_loss = self.model.full_loss(self.current_state + self.grad / self.L)
        #     print(next_loss)
        # self.current_state += self.grad / self.L
        # print("grad step loss: ", next_loss)

    def random_step(self):
        self.current_state += torch.normal(mean = 0., std = torch.abs(self.current_state*1e-3))
                
    def step(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            print("---------iter---------")
        else:
            print("---------init---------")
        h = self.update_h()
        
        with torch.no_grad():
            start = 0
            if self.iteration > 0:
                for P, V in zip(self.model.parameter_order, self.model.parameter_vector_len):
                    # print(self.model.name, P, "state", self.current_state[start:start + V], "h", h[start:start + V], "grad", self.grad[start:start + V])
                    start += V
            self.current_Y = self.model.full_sample(self.current_state + h).view(-1)
            if self.model.target.has_mask: # fixme something to do with the mask is a problem
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            print("nan loss, taking small grad step then will restart")
            self.grad_step()
            h = torch.zeros_like(self.current_state)
        elif self.iteration > 0:
            print("LM loss, best loss, L: ", loss.item(), np.min(self.loss_history[:-1]), self.L)
            rho = self.rho(np.min(self.loss_history[:-1]), loss, h) 
            print("rho: ", rho.item())
            if rho > self.epsilon4:
                print("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / self.Ldn)
                self._count_reject = 0
                if 0 < ((np.min(self.loss_history[:-1]) - loss) / loss) < 1e-6:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
            elif self._count_reject < 3:
                print("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1
                return
            # elif self._count_reject > 8:
            #     print("reject > 8 taking random step, last hope")
            #     self._count_reject += 1
            #     self.random_step()
            # elif self._count_reject > 6:
            #     print("reject > 6 taking grad step")
            #     self._count_reject += 1
            #     self.grad_step()
            #     h = torch.zeros_like(self.current_state)
            else:
                print("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1

        if self.J is None or self.iteration < 2 or self._count_reject >= 4 or rho == 0:
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
        self.grad_only = False

        try:
            while True:

                self.step()
            
                if self._count_finish >= 3:
                    self.message = self.message + "success"
                    break
                elif self.L >= (1e7 - 1) and self._count_reject >= 12:
                    self.message = self.message + "fail reject 12 in a row"
                    break
                elif self.iteration >= self.max_iter:
                    self.message = self.message + f"fail max iterations reached: {self.iteration}"
                    break
                elif not torch.all(torch.isfinite(self.current_state)):
                    self.message = self.message + "fail non-finite step taken"
                    break
        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"
            
        self.model.step(torch.tensor(self.res()))
        self.model.finalize()

        # set the uncertainty for each parameter
        cov = self.covariance_matrix()
        self.model.set_uncertainty(torch.diag(cov), uncertainty_as_representation = True)
        
        return self
            
    def update_h(self):

        if self.grad_only or self._count_reject > 8:
            print("pure grad")
            return self.grad / self.L
        count_reject = 0
        h = torch.zeros(len(self.current_state))
        if self.iteration == 0:
            return h
        while count_reject < 4:
            # Sometimes the hesian + lambda matrix is singular, sometimes that can be fixed by giving lambda a boost.
            try:
                with torch.no_grad():
                    h = torch.linalg.solve(self.hess + self.L*torch.sqrt(torch.diag(self.hess))*torch.eye(len(self.grad)), self.grad)
                break
            except Exception as e:
                print("reject err: ", e)
                print("WARNING: will massage Hessian to continue, results may not converge")
                self.hess *= torch.eye(len(self.grad))*0.9 + 0.1
                self.hess += torch.eye(len(self.grad))
                self.L = min(1e7, self.L * self.Lup)
                count_reject += 1
        return h
    
    def update_J_AD(self, h):
        self.J = self.model.jacobian(self.current_state + h).view(-1,len(self.current_state))
        if self.model.target.has_mask:
            self.J[self.mask] = 0.
            
    @torch.no_grad()
    def update_J_Broyden(self, h, Yp, Yph):
        self.J += torch.outer(Yph - Yp - torch.matmul(self.J, h),h) / torch.linalg.norm(h)
        if self.model.target.has_mask:
            self.J[self.mask] = 0.

    @torch.no_grad()
    def update_hess(self):
        if self.W is None:
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W),-1)*self.J)

    @torch.no_grad()
    def covariance_matrix(self):
        return torch.linalg.inv(self.hess)
            
    @torch.no_grad()
    def update_grad(self, Yph):
        if self.W is None:
            self.grad = torch.matmul(self.J.T, (self.Y - Yph))
        else:
            self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))
            
    @torch.no_grad()
    def rho(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess)) * h) + self.grad))
