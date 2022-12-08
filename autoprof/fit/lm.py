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
        
        self.epsilon4 = kwargs.get("epsilon4", 0.1)
        self.Lup = kwargs.get("Lup", 7.)
        self.Ldn = kwargs.get("Ldn", 5.)
        self.L = kwargs.get("L0", 1.)
        self.method = kwargs.get("method", 1)
        
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
        self.decision_history = []
        self.rho_history = []
        
    def grad_step(self):

        print(self.current_state)
        print(self.model.full_loss(self.current_state))
        grad_res = Grad(self.model, self.current_state, max_iter = 20, optim_kwargs = {"lr": 1e-3}).fit()
        self.current_state = torch.tensor(grad_res.lambda_history[np.nanargmin(grad_res.loss_history)], dtype = self.model.dtype, device = self.model.device)
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
                
    def step_method1(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            print("---------iter---------")
        else:
            print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        h = self.update_h_v2()
        
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
            print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.L = min(1e9, self.L * self.Lup)
            return
        elif self.iteration > 0:
            print("LM loss, best loss, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            rho = self.rho_3(np.nanmin(self.loss_history[:-1]), loss, h)
            self.rho_history.append(rho)
            print("rho: ", rho.item())
            if rho > self.epsilon4:
                print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / self.Ldn)
                self._count_reject = 0
                if 0 < ((np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
            elif self._count_reject == 8:
                print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = min(1e-2, self.L / self.Lup**8)
                self._count_reject += 1                
            else:
                print("reject")
                self.decision_history.append("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1
                return
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or rho < 0.1 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            self.update_J_AD()
            print("full jac")
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)
            print("Broyden jac")

        self.update_hess()
        self.update_grad(self.current_Y)
        self.iteration += 1

    def step_method2(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            print("---------iter---------")
        else:
            print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        if self.iteration == 1:
            self.L = self.L * np.max(torch.diag(self.hess).detach().cpu().numpy())
        h = self.update_h_v1()
        
        with torch.no_grad():
            start = 0
            if self.iteration > 0:
                for P, V in zip(self.model.parameter_order, self.model.parameter_vector_len):
                    # print(self.model.name, P, "state", self.current_state[start:start + V], "h", h[start:start + V], "grad", self.grad[start:start + V])
                    start += V
            self.current_Y = self.model.full_sample(self.current_state + h).view(-1)
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.L = min(1e9, self.L * self.Lup)
            return
        elif self.iteration > 0:
            print("LM loss, best loss, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            alpha = torch.dot(self.grad, h) 
            alpha = alpha / ((loss - np.nanmin(self.loss_history[:-1]))/2 + 2*alpha)
            self.current_Y = self.model.full_sample(self.current_state + alpha*h).view(-1)
            if self.model.target.has_mask:
                alpha_loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                alpha_loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
            rho = self.rho_2(np.nanmin(self.loss_history[:-1]), alpha_loss, h)
            self.rho_history.append(rho)
            print("rho: ", rho.item())
            if rho > self.epsilon4:
                print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / (1+alpha))
                self._count_reject = 0
                if 0 < ((np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
            elif self._count_reject == 8:
                print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = 1e-2
                self._count_reject += 1                
            else:
                print("reject")
                self.decision_history.append("reject")
                self.L = min(1e9, self.L + np.abs(alpha_loss - np.nanmin(self.loss_history[:-1])) / (2*alpha))
                self._count_reject += 1
                return
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or rho < 0.1 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            self.update_J_AD()
            print("full jac")
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)
            print("Broyden jac")

        self.update_hess()
        self.update_grad(self.current_Y)
        self.iteration += 1
        
    def step_method3(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            print("---------iter---------")
        else:
            print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        if self.iteration == 1:
            self.L = self.L * np.max(torch.diag(self.hess).detach().cpu().numpy())
        h = self.update_h_v1()
        
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
            print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.L = min(1e9, self.L * self.Lup)
            return
        elif self.iteration > 0:
            print("LM loss, best loss, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            rho = self.rho_2(np.nanmin(self.loss_history[:-1]), loss, h)
            self.rho_history.append(rho)
            print("rho: ", rho.item())
            if rho > self.epsilon4:
                print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / 3)
                self.v = 2.
                self._count_reject = 0
                if 0 < ((np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
            elif self._count_reject == 8:
                print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = 1e-2
                self._count_reject += 1                
            else:
                print("reject")
                self.decision_history.append("reject")
                self.L = min(1e9, self.L * self.v)
                self.v *= 2
                self._count_reject += 1
                return
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or rho < 0.1 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            self.update_J_AD()
            print("full jac")
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)
            print("Broyden jac")

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
        self.v = 1.
        
        try:
            while True:

                if self.method == 3:
                    self.step_method3()
                elif self.method == 2:
                    self.step_method2()
                else:
                    self.step_method1()
                    
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
        self.update_J_AD()
        self.update_hess()
        cov = self.covariance_matrix()
        self.model.set_uncertainty(torch.sqrt(2*torch.abs(torch.diag(cov))), uncertainty_as_representation = True)
        
        return self
            
    @torch.no_grad()
    def update_h_v1(self):
        if self.iteration == 0:
            return torch.zeros_like(self.current_state)
        return torch.linalg.solve(self.hess + self.L*torch.eye(len(self.current_state)), self.grad)
    @torch.no_grad()
    def update_h_v2(self):

        count_reject = 0
        h = torch.zeros_like(self.current_state)
        if self.iteration == 0:
            return h
        while count_reject < 4:
            # Sometimes the hesian + lambda matrix is singular, sometimes that can be fixed by giving lambda a boost.
            try:
                h = torch.linalg.solve(self.hess + self.L*torch.abs(torch.diag(self.hess))*torch.eye(len(self.grad)), self.grad)
                break
            except Exception as e:
                print("reject err: ", e)
                print("WARNING: will massage Hessian to continue, results may not converge")
                self.hess *= torch.eye(len(self.grad))*0.9 + 0.1
                self.hess += torch.eye(len(self.grad))
                self.L = min(1e7, self.L * self.Lup)
                count_reject += 1
        return h
    
    def update_J_AD(self):
        self.J = self.model.jacobian(self.current_state).view(-1,len(self.current_state))
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
        try:
            return torch.linalg.inv(self.hess)
        except:
            print("WARNING: Hessian is singular, likely at least one model is non-physical. Will massage Hessian to continue but results should be inspected.")
            self.hess += torch.eye(len(self.grad))*(torch.diag(self.hess) == 0)
            return torch.linalg.inv(self.hess)
            
    @torch.no_grad()
    def update_grad(self, Yph):
        if self.W is None:
            self.grad = torch.matmul(self.J.T, (self.Y - Yph))
        else:
            self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))
            
    @torch.no_grad()
    def rho_1(self, Xp, Xph, h):
        update = self.Y - self.current_Y - torch.matmul(self.J,h)
        if self.model.target.has_mask:
            return self.ndf*(Xp - Xph) / abs(self.ndf*Xp - torch.dot(update,(self.W * update)))
        else:
            return self.ndf*(Xp - Xph) / abs(self.ndf*Xp - torch.dot(update,update))
    @torch.no_grad()
    def rho_2(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * h + self.grad))
    @torch.no_grad()
    def rho_3(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess)) * h) + self.grad))