# Levenberg-Marquardt algorithm
import os
import torch
import numpy as np
from time import time
from .base import BaseOptimizer

__all__ = ["LM"]

@torch.no_grad()
@torch.jit.script
def Broyden_step(J, h, Yp, Yph):
    delta = torch.matmul(J, h)
    # avoid constructing a second giant jacobian matrix, instead go one row at a time
    for j in range(J.shape[1]):
        J[:,j] += (Yph - Yp - delta) * h[j] / torch.linalg.norm(h)
    return J

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

    Once a Hessian H and gradient G have been determined, the update
    step is defined as h which is the solution to the linear equation:

    (H + L*I)h = G

    where L is the Levenberg-Marquardt damping parameter and I is the
    identity matrix. For small L this is just the Newton's method, for
    large L this is just a small gradient descent step (approximately
    h = grad/L). The three methods implimented come from Gavin
    2019. Note that in method 1 the identity matrix is replace with
    diag(H) so that each parameter is scaled by its second derivative.

    Parameters:
        model: and AutoProf_Model object with which to perform optimization [AutoProf_Model object]
        initial_state: optionally, and initial state for optimization [torch.Tensor]
        method: optimization method to use for the update step [int]
        epsilon4: approximation accuracy requirement, for any rho < epsilon4 the step will be rejected
        epsilon5: numerical stability factor, added to the diagonal of the Hessian
        L0: initial value for L factor in (H +L*I)h = G
        Lup: method1 amount to increase L when rejecting an update step
        Ldn: amount to decrease L when accetping an update step

    """
    
    def __init__(self, model, initial_state = None, max_iter = 100, **kwargs):
        super().__init__(model, initial_state, max_iter = max_iter, **kwargs)
        
        self.epsilon4 = kwargs.get("epsilon4", 0.1)
        self.epsilon5 = kwargs.get("epsilon5", 1e-8)
        self.Lup = kwargs.get("Lup", 5.)
        self.Ldn = kwargs.get("Ldn", 3.)
        self.L = kwargs.get("L0", 1.)
        self.method = kwargs.get("method", 0)
        
        self.Y = self.model.target[self.model.window].flatten("data")
        #        1 / sigma^2
        self.W = 1. / self.model.target[self.model.window].flatten("variance") if model.target.has_variance else None
        #          # pixels      # parameters
        self.ndf = len(self.Y) - len(self.current_state)
        self.J = None
        self.current_Y = None
        self.prev_Y = [None, None]
        if self.model.target.has_mask:
            self.mask = self.model.target[self.model.window].flatten("mask")
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(self.mask)
        self.L_history = []
        self.decision_history = []
        self.rho_history = []

    def step_method0(self, current_state = None):
        """
        same as method one, except that the off diagonal elements are scaled by 1/(1+L) making the move to pure gradient descent faster and better behaved
        """
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            if self.verbose > 0:
                print("---------iter---------")
        else:
            if self.verbose > 0:
                print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        h = self.update_h_v3()
        if self.verbose > 2:
            print("h: ", h.detach().cpu().numpy())
        with torch.no_grad():
            self.current_Y = self.model.full_sample(self.current_state + h, as_representation = True, override_locked = False, flatten = True)
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            if self.verbose > 0:
                print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.iteration += 1
            #self.undo_step()
            self.L = min(1e9, self.L*self.Lup)
            return
        elif self.iteration > 0:
            rho = self.rho_3(np.nanmin(self.loss_history[:-1]), loss, h)
            if self.verbose > 1:
                print("LM loss, best loss, loss diff, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            elif self.verbose > 0 and rho > self.epsilon4:
                print("LM loss", loss.item())
            self.rho_history.append(rho)
            if self.verbose > 1:
                print("rho: ", rho.item())
                
            if rho > self.epsilon4:
                if self.verbose > 0:
                    print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / self.Ldn)
                self._count_reject = 0
                if 0 < (self.ndf * (np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
            elif self._count_reject == 8:
                if self.verbose > 1:
                    print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = min(1e-2, self.L / self.Lup**8)
                self._count_reject += 1                
            else:
                if self.verbose > 1:
                    print("reject")
                self.decision_history.append("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1
                return    
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or "reset" in self.decision_history[-2:] or rho < self.epsilon4 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            if self.verbose > 1:
                print("full jac")
            self.update_J_AD()
        else:
            if self.verbose > 1:
                print("Broyden jac")
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)

        self.update_hess()
        self.update_grad(self.prev_Y[1])
        self.iteration += 1
        
    def step_method1(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            if self.verbose > 0:
                print("---------iter---------")
        else:
            if self.verbose > 0:
                print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        h = self.update_h_v2()
        if self.verbose > 2:
            print("h: ", h.detach().cpu().numpy())
        
        with torch.no_grad():
            self.current_Y = self.model.full_sample(self.current_state + h, as_representation = True, override_locked = False, flatten = True)
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            if self.verbose > 0:
                print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.L = min(1e9, self.L*self.Lup)
            return
        elif self.iteration > 0:
            rho = self.rho_3(np.nanmin(self.loss_history[:-1]), loss, h)
            if self.verbose > 1:
                print("LM loss, best loss, loss diff, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            elif self.verbose > 0 and rho > self.epsilon4:
                print("LM loss", loss.item())
            self.rho_history.append(rho)
            if self.verbose > 1:
                print("rho: ", rho.item())
                
            if rho > self.epsilon4:
                if self.verbose > 0:
                    print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / self.Ldn)
                self._count_reject = 0
                if 0 < (self.ndf * (np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
            elif self._count_reject == 8:
                if self.verbose > 1:
                    print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = min(1e-2, self.L / self.Lup**8)
                self._count_reject += 1                
            else:
                if self.verbose > 1:
                    print("reject")
                self.decision_history.append("reject")
                self.L = min(1e9, self.L * self.Lup)
                self._count_reject += 1
                return
            
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or "reset" in self.decision_history[-2:] or rho < self.epsilon4 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            if self.verbose > 1:
                print("full jac")
            self.update_J_AD()
        else:
            if self.verbose > 1:
                print("Broyden jac")
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)

        self.update_hess()
        self.update_grad(self.prev_Y[1])
        self.iteration += 1        

    def step_method2(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            if self.verbose > 0:
                print("---------iter---------")
        else:
            if self.verbose > 0:
                print("---------init---------")
        # if self.iteration > 6:
        #     if self._count_reject >= 6:
        #         self.L = self.L_history[-6:][np.argmax(np.abs(self.rho_history[-6:]))] * np.exp(np.random.normal(loc = 0, scale = 1))
        if self.iteration == 1:
            self.L = self.L * np.max(torch.diag(self.hess).detach().cpu().numpy())
        h = self.update_h_v1()
        
        with torch.no_grad():
            self.current_Y = self.model.full_sample(self.current_state + h, as_representation = True, override_locked = False, flatten = True)
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            if self.verbose > 0:
                print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.L = min(1e9, self.L * self.Lup)
            return
        elif self.iteration > 0:
            if self.verbose > 1:
                print("LM loss, best loss, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            elif self.verbose > 0:
                print("LM loss", loss.item())
            alpha = torch.dot(self.grad, h) 
            alpha = alpha / ((loss - np.nanmin(self.loss_history[:-1]))/2 + 2*alpha)
            self.current_Y = self.model.full_sample(self.current_state + alpha*h).view(-1)
            if self.model.target.has_mask:
                alpha_loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                alpha_loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
            rho = self.rho_2(np.nanmin(self.loss_history[:-1]), alpha_loss, h)
            self.rho_history.append(rho)
            if self.verbose > 1:
                print("rho: ", rho.item())
            if rho > self.epsilon4:
                if self.verbose > 0:
                    print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L = max(1e-9, self.L / (1+alpha))
                self._count_reject = 0
                if 0 < ((np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
            elif self._count_reject == 6:
                if self.verbose > 0:
                    print("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = 1e-2
                self._count_reject += 1                
            else:
                if self.verbose > 0:
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
            if self.verbose > 1:
                print("full jac")
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)
            if self.verbose > 1:
                print("Broyden jac")

        self.update_hess()
        self.update_grad(self.prev_Y[1])
        self.iteration += 1
        
    def step_method3(self, current_state = None):
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            if self.verbose > 0:
                print("---------iter---------")
        else:
            if self.verbose > 0:
                print("---------init---------")
                
        if self.iteration > 0:
            self.L = self.Li * np.max(torch.diag(self.hess).detach().cpu().numpy())
        h = self.update_h_v1()
        
        with torch.no_grad():
            self.current_Y = self.model.full_sample(self.current_state + h, as_representation = True, override_locked = False, flatten = True)
            if self.model.target.has_mask: 
                loss = torch.sum(((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W))[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 if self.W is None else ((self.Y - self.current_Y)**2 * self.W)) / self.ndf
        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if not torch.isfinite(loss):
            if self.verbose > 0:
                print("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.Li = min(1e9, self.Li * v)
            return
        elif self.iteration > 0:
            rho = self.rho_2(np.nanmin(self.loss_history[:-1]), loss, h)
            if self.verbose > 1:
                print("LM loss, best loss, L: ", loss.item(), np.nanmin(self.loss_history[:-1]), np.nanmin(self.loss_history[:-1]) - loss.item(), self.L)
            elif self.verbose > 0 and rho > self.epsilon4:
                print("LM loss: ", loss.item())
            self.rho_history.append(rho)
            if self.verbose > 1:
                print("rho: ", rho.item())
            if rho > self.epsilon4:
                if self.verbose > 1:
                    print("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.Li = max(1e-9, self.Li / 5)
                self.v = 2.
                self._count_reject = 0
                if 0 < (self.ndf*(np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
            elif self._count_reject == 8:# fixme state not fully reset somehow
                if self.verbose > 1:
                    print("reject, resetting jacobian")
                self.decision_history.append("reset")
                self.Li = 1.
                self.v = 1.
                self._count_reject += 1
            else:
                if self.verbose > 1:
                    print("reject")
                self.decision_history.append("reject")
                self.Li = min(1e9, self.Li * self.v)
                self.v *= 2.
                self._count_reject += 1
                return
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if self.J is None or self.iteration < 2 or "reset" in self.decision_history[-2:] or rho < self.epsilon4 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            self.update_J_AD()
            if self.verbose > 1:
                print("full jac")
        else:
            self.update_J_Broyden(h, self.prev_Y[0], self.prev_Y[1])
            if self.verbose > 1:
                print("Broyden jac")

        self.update_hess()
        self.update_grad(self.prev_Y[1])
        self.iteration += 1
        
        
    def fit(self):

        self.iteration = 0
        self._count_reject = 0
        self._count_finish = 0
        self.grad_only = False
        if self.method == 3:
            self.v = 1.
            self.Li = self.L
        
        try:
            while True:
                if self.verbose > 0:
                    print("L: ", self.L)
                    
                if self.method == 3:
                    self.step_method3()
                elif self.method == 2:
                    self.step_method2()
                if self.method == 1:
                    self.step_method1()                    
                else:
                    self.step_method0()

                if self.save_steps is not None and self.decision_history[-1] == "accept":
                    self.model.save(os.path.join(self.save_steps, f"{self.model.name}_Iteration_{self.iteration:03d}.yaml"))

                lam, L, loss = self.progress_history()
                    
                if self.decision_history.count("accept") > 2 and self.decision_history[-1] == "accept" and L[-1] < 0.1 and ((loss[-2] - loss[-1])/loss[-1]) < (self.relative_tolerance/100):
                    self.message = self.message + "success"
                    break
                elif self.L >= (1e9 - 1) and self._count_reject >= 12 and not self.take_low_rho_step():
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

        if "fail" in self.message and self._count_finish > 0:
            self.message = self.message + ". likely converged to numerical precision and could not make a better step, this is probably ok."
        self.model.set_parameters(torch.tensor(self.res(), dtype = self.model.dtype, device = self.model.device), as_representation = True, override_locked = False)
        self.model.finalize()

        # set the uncertainty for each parameter
        self.update_J_AD()
        self.update_hess()
        cov = self.covariance_matrix()
        self.model.set_uncertainty(torch.sqrt(2*torch.abs(torch.diag(cov))), as_representation = True, override_locked = False)
        
        return self

    @torch.no_grad()
    def undo_step(self):
        print("undoing step, trying to recover")
        assert self.decision_history.count("accept") >= 2, "cannot undo with not enough accepted steps, retry with new parameters"
        assert len(self.decision_history) == len(self.lambda_history)
        assert len(self.decision_history) == len(self.L_history)
        found_accept = False
        for i in reversed(range(len(self.decision_history))):
            if not found_accept and self.decision_history[i] == "accept":
                found_accept = True
                continue
            if self.decision_history[i] != "accept":
                continue
            self.current_state = torch.tensor(self.lambda_history[i], dtype = self.model.dtype, device = self.model.device)
            self.L = self.L_history[i] * self.Lup

    
    def take_low_rho_step(self):
        
        for i in reversed(range(len(self.decision_history))):
            if self.decision_history[i] == "accept":
                return False
            if self.rho_history[i] is not None and self.rho_history[i] > 0:
                if self.verbose > 0:
                    print("taking a low rho step for some progress: ", self.rho_history[i])
                self.current_state = torch.tensor(self.lambda_history[i], dtype = self.model.dtype, device = self.model.device)
                self.L = self.L_history[i]
                
                self.loss_history.append(self.loss_history[i])
                self.L_history.append(self.L)
                self.lambda_history.append(np.copy((self.current_state).detach().cpu().numpy()))
                self.decision_history.append("low rho accept")
                self.rho_history.append(self.rho_history[i])

                with torch.no_grad():
                    Y = self.model.full_sample(self.current_state, as_representation = True, override_locked = False, flatten = True)
                    self.prev_Y[0] = self.prev_Y[1]
                    self.prev_Y[1] = Y
                self.update_J_AD()
                self.update_hess()
                self.update_grad(self.prev_Y[1])
                self.iteration += 1
                self.count_reject = 0
                return True
            
    @torch.no_grad()
    def update_h_v1(self):
        if self.iteration == 0:
            return torch.zeros_like(self.current_state)
        return torch.linalg.solve(self.hess + self.L*torch.eye(len(self.current_state), dtype = self.model.dtype, device = self.model.device), self.grad)
    @torch.no_grad()
    def update_h_v2(self):

        count_reject = 0
        h = torch.zeros_like(self.current_state)
        if self.iteration == 0:
            return h
        h = torch.linalg.solve(self.hess * (1 + self.L*torch.eye(len(self.grad), dtype = self.model.dtype, device = self.model.device)), self.grad)
        return h
    
    @torch.no_grad()
    def update_h_v3(self):
        h = torch.zeros_like(self.current_state)
        if self.iteration == 0:
            return h
        h = torch.linalg.solve((self.hess + 1e-3*self.L*torch.eye(len(self.grad), dtype = self.model.dtype, device = self.model.device)) * (1 + self.L*torch.eye(len(self.grad), dtype = self.model.dtype, device = self.model.device))**2/(1 + self.L), self.grad)
        return h
    
    def update_J_AD(self):
        del self.J
        if "cpu" not in self.model.device:
            torch.cuda.empty_cache()
        self.J = self.model.jacobian(torch.clone(self.current_state).detach(), as_representation = True, override_locked = False, flatten = True)
        if self.model.target.has_mask:
            self.J[self.mask] = 0.
            
    @torch.no_grad()
    def update_J_Broyden(self, h, Yp, Yph):
        self.J = Broyden_step(self.J, h, Yp, Yph)
        if self.model.target.has_mask:
            self.J[self.mask] = 0.

    @torch.no_grad()
    def update_hess(self):
        if self.W is None:
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W),-1)*self.J)
        self.hess += self.epsilon5 * torch.eye(len(self.current_state), dtype = self.model.dtype, device = self.model.device)
            
    @torch.no_grad()
    def covariance_matrix(self):
        try:
            return torch.linalg.inv(self.hess)
        except:
            print("WARNING: Hessian is singular, likely at least one model is non-physical. Will massage Hessian to continue but results should be inspected.")
            self.hess += torch.eye(len(self.grad), dtype = self.model.dtype, device = self.model.device)*(torch.diag(self.hess) == 0)
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
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess) - self.epsilon5) * h) + self.grad))

    def progress_history(self):
        lambdas = []
        Ls = []
        losses = []

        for l in range(len(self.decision_history)):
            if self.decision_history[l] == "accept":
                lambdas.append(self.lambda_history[l])
                Ls.append(self.L_history[l])
                losses.append(self.loss_history[l])
        return lambdas, Ls, losses
