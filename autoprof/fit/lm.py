# Levenberg-Marquardt algorithm
import os
import torch
import numpy as np
from time import time
from .base import BaseOptimizer
from .. import AP_config

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
    h = grad/L). The method implimented is modified from Gavin 2019.

    Parameters:
        model: and AutoProf_Model object with which to perform optimization [AutoProf_Model object]
        initial_state: optionally, and initial state for optimization [torch.Tensor]
        epsilon4: approximation accuracy requirement, for any rho < epsilon4 the step will be rejected
        epsilon5: numerical stability factor, added to the diagonal of the Hessian
        L0: initial value for L factor in (H +L*I)h = G
        Lup: amount to increase L when rejecting an update step
        Ldn: amount to decrease L when accetping an update step

    """
    
    def __init__(self, model, initial_state = None, max_iter = 100, **kwargs):
        super().__init__(model, initial_state, max_iter = max_iter, **kwargs)
        
        self.epsilon4 = kwargs.get("epsilon4", 0.1)
        self.epsilon5 = kwargs.get("epsilon5", 1e-8)
        self.Lup = kwargs.get("Lup", 11.)
        self.Ldn = kwargs.get("Ldn", 9.)
        self.L = kwargs.get("L0", 1.)
        self.use_broyden = kwargs.get("use_broyden", False)
        
        self.Y = self.model.target[self.model.window].flatten("data")
        #        1 / sigma^2
        self.W = 1. / self.model.target[self.model.window].flatten("variance") if model.target.has_variance else 1.
        #          # pixels      # parameters
        self.ndf = len(self.Y) - len(self.current_state)
        self.J = None
        self.full_jac = False
        self.current_Y = None
        self.prev_Y = [None, None]
        if self.model.target.has_mask:
            self.mask = self.model.target[self.model.window].flatten("mask")
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(self.mask)
        self.L_history = []
        self.decision_history = []
        self.rho_history = []
        self._count_grad_step = 0
        self._count_converged = 0

    def L_up(self, Lup = None):
        if Lup is None:
            Lup = self.Lup
        self.L = min(1e9, self.L*Lup)
    def L_dn(self, Ldn = None):
        if Ldn is None:
            Ldn = self.Ldn
        self.L = max(1e-9, self.L/Ldn)
        
    @torch.no_grad()
    def grad_step(self):
        L = 0.1
        self.iteration += 1
        self._count_grad_step += 1
        if self.verbose > 1:
            AP_config.ap_logger.info(f"taking grad step. Loss to beat: {np.nanmin(self.loss_history[:-1])}")
        for count in range(20):
            Y = self.model(parameters = self.current_state + self.grad*L, as_representation = True, override_locked = False).flatten("data")
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - Y)**2 * self.W)[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - Y)**2 * self.W) / self.ndf
            if not torch.isfinite(loss):
                L /= 10
                continue
            if self.verbose > 1:
                AP_config.ap_logger.info(f"grad step loss: {loss.item()}, L: {L}")
            if np.nanmin(self.loss_history[:-1]) > loss.item():
                self.loss_history.append(loss.detach().cpu().item())
                self.L = 1.
                self.L_history.append(self.L)
                self.current_state += self.grad*L
                self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
                self.decision_history.append("accept grad")
                if self.verbose > 0:
                    AP_config.ap_logger.info("accept grad")
                self.rho_history.append(1.)
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(Y)
                break
            elif np.abs(np.nanmin(self.loss_history[:-1]) - loss.item()) < (self.relative_tolerance * 1e-3) and L < 1e-5:
                self.loss_history.append(loss.detach().cpu().item())
                self.L = 1.
                self.L_history.append(self.L)
                self.current_state += self.grad*L
                self.lambda_history.append(np.copy(self.current_state.detach().cpu().numpy()))
                self.decision_history.append("accept bad grad")
                if self.verbose > 0:
                    AP_config.ap_logger.info("accept bad grad")
                self.rho_history.append(1.)                
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(Y)
                break
            else:
                L /= 10
                continue
        else:
            raise RuntimeError("Unable to take gradient step! LM has found itself in a very bad place of parameter space, try adjusting initial parameters")
        
    def step(self, current_state = None):
        """
        Levenberg-Marquardt update step
        """
        if current_state is not None:
            self.current_state = current_state

        if self.iteration > 0:
            if self.verbose > 0:
                AP_config.ap_logger.info("---------iter---------")
        else:
            if self.verbose > 0:
                AP_config.ap_logger.info("---------init---------")
                
        h = self.update_h()
        if self.verbose > 1:
            AP_config.ap_logger.debug(f"h: {h.detach().cpu().numpy()}")
        with torch.no_grad():
            self.current_Y = self.model(parameters = self.current_state + h, as_representation = True, override_locked = False).flatten("data")
            if self.model.target.has_mask:
                loss = torch.sum(((self.Y - self.current_Y)**2 * self.W)[torch.logical_not(self.mask)]) / self.ndf
            else:
                loss = torch.sum((self.Y - self.current_Y)**2 * self.W) / self.ndf
        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(np.copy((self.current_state + h).detach().cpu().numpy()))
        
        if self.iteration > 0 and not torch.isfinite(loss):
            if self.verbose > 0:
                AP_config.ap_logger.warning("nan loss")
            self.decision_history.append("nan")
            self.rho_history.append(None)
            self._count_reject += 1
            self.iteration += 1
            self.L_up()
            return
        elif self.iteration > 0:
            rho = self.rho(np.nanmin(self.loss_history[:-1]), loss, h)
            if self.verbose > 1:
                AP_config.ap_logger.debug(f"LM loss: {loss.item()}, best loss: {np.nanmin(self.loss_history[:-1])}, loss diff: {np.nanmin(self.loss_history[:-1]) - loss.item()}, L: {self.L}")
            elif self.verbose > 0 and rho > self.epsilon4:
                AP_config.ap_logger.info(f"LM loss: {loss.item()}")
            self.rho_history.append(rho)
            if self.verbose > 1:
                AP_config.ap_logger.debug(f"rho: {rho.item()}")
                
            if rho > self.epsilon4:
                if self.verbose > 0:
                    AP_config.ap_logger.info("accept")
                self.decision_history.append("accept")
                self.prev_Y[0] = self.prev_Y[1]
                self.prev_Y[1] = torch.clone(self.current_Y)
                self.current_state += h
                self.L_dn()
                self._count_reject = 0
                if 0 < (self.ndf * (np.nanmin(self.loss_history[:-1]) - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
            elif self._count_reject == 4:
                if self.verbose > 0:
                    AP_config.ap_logger.info("reject, resetting jacobian")
                self.decision_history.append("reject")
                self.L = min(1e-2, self.L / self.Lup**4)
                self._count_reject += 1                
            else:
                if self.verbose > 0:
                    AP_config.ap_logger.info("reject")
                self.decision_history.append("reject")
                self.L_up()
                self._count_reject += 1
                return    
        else:
            self.decision_history.append("init")
            self.rho_history.append(None)

        if (not self.use_broyden) or self.J is None or self.iteration < 2 or "reset" in self.decision_history[-2:] or rho < self.epsilon4 or self._count_reject > 0 or self.iteration >= (2 * len(self.current_state)) or self.decision_history[-1] == "nan":
            if self.verbose > 1:
                AP_config.ap_logger.debug("full jac")
            self.update_J_AD()
        else:
            if self.verbose > 1:
                AP_config.ap_logger.debug("Broyden jac")
            self.update_J_Broyden(h, self.prev_Y[0], self.current_Y)

        self.update_hess()
        self.update_grad(self.prev_Y[1])
        self.iteration += 1        
        
    def fit(self):

        self.iteration = 0
        self._count_reject = 0
        self._count_finish = 0
        self.grad_only = False

        start_fit = time()
        try:
            while True:
                if self.verbose > 0:
                    AP_config.ap_logger.info(f"L: {self.L}")

                # take LM step
                self.step()

                # Save the state of the model
                if self.save_steps is not None and self.decision_history[-1] == "accept":
                    self.model.save(os.path.join(self.save_steps, f"{self.model.name}_Iteration_{self.iteration:03d}.yaml"))

                lam, L, loss = self.progress_history()

                # Check for convergence
                if self.decision_history.count("accept") > 2 and self.decision_history[-1] == "accept" and L[-1] < 0.1 and ((loss[-2] - loss[-1])/loss[-1]) < (self.relative_tolerance/10):
                    self._count_grad_step = 0
                    self._count_converged += 1
                elif self._count_grad_step >= 5:
                    self.message = self.message + "success by immobility, unable to find improvement either converged or bad area of parameter space."
                    break
                elif self.iteration >= self.max_iter:
                    self.message = self.message + f"fail max iterations reached: {self.iteration}"
                    break
                elif not torch.all(torch.isfinite(self.current_state)):
                    self.message = self.message + "fail non-finite step taken"
                    break
                elif self.L >= (1e9 - 1) and self._count_reject >= 12 and not self.take_low_rho_step():
                    if not self.full_jac:
                        self.update_J_AD()
                        self.update_hess()
                        self.update_grad(self.prev_Y[1])
                    try:
                        self.grad_step()
                    except RuntimeError:
                        self.message = self.message + "fail by immobility, unable to find improvement or even small bad step"
                        break
                if self._count_converged >= 2:
                    self.message = self.message + "success"
                    break
                lam, L, loss = self.accept_history()
                if len(loss) >= 10:
                    loss10 = np.array(loss[-10:])
                    if np.all(np.abs((loss10[1:] - loss10[:-1]) / loss10[:-1]) < self.relative_tolerance):
                        self.message = self.message + "success"
                        break
        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"

            
        if self.message.startswith("fail") and self._count_finish > 0:
            self.message = self.message + ". possibly converged to numerical precision and could not make a better step."
        self.model.set_parameters(self.res(), as_representation = True, override_locked = False)
        if self.verbose > 1:
            AP_config.ap_logger.info("LM Fitting complete in {time() - start_fit} sec with message: self.message")
        # set the uncertainty for each parameter
        if self.use_broyden:
            self.update_J_AD()
            self.update_hess()
        cov = self.covariance_matrix()
        self.model.set_uncertainty(torch.sqrt(2*torch.abs(torch.diag(cov))), as_representation = True, override_locked = False)
        
        return self

    @torch.no_grad()
    def undo_step(self):
        AP_config.ap_logger.info("undoing step, trying to recover")
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
            self.current_state = torch.tensor(self.lambda_history[i], dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            self.L = self.L_history[i] * self.Lup
    
    def take_low_rho_step(self):
        
        for i in reversed(range(len(self.decision_history))):
            if "accept" in self.decision_history[i]:
                return False
            if self.rho_history[i] is not None and self.rho_history[i] > 0:
                if self.verbose > 0:
                    AP_config.ap_logger.info(f"taking a low rho step for some progress: {self.rho_history[i]}")
                self.current_state = torch.tensor(self.lambda_history[i], dtype = AP_config.ap_dtype, device = AP_config.ap_device)
                self.L = self.L_history[i]
                
                self.loss_history.append(self.loss_history[i])
                self.L_history.append(self.L)
                self.lambda_history.append(np.copy((self.current_state).detach().cpu().numpy()))
                self.decision_history.append("low rho accept")
                self.rho_history.append(self.rho_history[i])

                with torch.no_grad():
                    Y = self.model(parameters = self.current_state, as_representation = True, override_locked = False).flatten("data")
                    self.prev_Y[0] = self.prev_Y[1]
                    self.prev_Y[1] = Y
                self.update_J_AD()
                self.update_hess()
                self.update_grad(self.prev_Y[1])
                self.iteration += 1
                self.count_reject = 0
                return True
            
    @torch.no_grad()
    def update_h(self):
        h = torch.zeros_like(self.current_state)
        if self.iteration == 0:
            return h
        h = torch.linalg.solve((self.hess + 1e-3*self.L*torch.eye(len(self.grad), dtype = AP_config.ap_dtype, device = AP_config.ap_device)) * (1 + self.L*torch.eye(len(self.grad), dtype = AP_config.ap_dtype, device = AP_config.ap_device))**2/(1 + self.L), self.grad)
        return h
    
    def update_J_AD(self):
        del self.J
        if "cpu" not in AP_config.ap_device:
            torch.cuda.empty_cache()
        self.J = self.model.jacobian(torch.clone(self.current_state).detach(), as_representation = True, override_locked = False, flatten = True)
        if self.model.target.has_mask:
            self.J[self.mask] = 0.
        self.full_jac = True
            
    @torch.no_grad()
    def update_J_Broyden(self, h, Yp, Yph):
        self.J = Broyden_step(self.J, h, Yp, Yph)
        if self.model.target.has_mask:
            self.J[self.mask] = 0.
        self.full_jac = False

    @torch.no_grad()
    def update_hess(self):
        if isinstance(self.W, float):
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W),-1)*self.J)
        self.hess += self.epsilon5 * torch.eye(len(self.current_state), dtype = AP_config.ap_dtype, device = AP_config.ap_device)
            
    @torch.no_grad()
    def covariance_matrix(self):
        try:
            return torch.linalg.inv(self.hess)
        except:
            AP_config.ap_logger.warning("WARNING: Hessian is singular, likely at least one model is non-physical. Will massage Hessian to continue but results should be inspected.")
            self.hess += torch.eye(len(self.grad), dtype = AP_config.ap_dtype, device = AP_config.ap_device)*(torch.diag(self.hess) == 0)
            return torch.linalg.inv(self.hess)
            
    @torch.no_grad()
    def update_grad(self, Yph):
        self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))
            
    @torch.no_grad()
    def rho(self, Xp, Xph, h):
        return self.ndf*(Xp - Xph) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(self.hess) - self.epsilon5) * h) + self.grad))

    def accept_history(self):
        lambdas = []
        Ls = []
        losses = []

        for l in range(len(self.decision_history)):
            if "accept" in self.decision_history[l] and np.isfinite(self.loss_history[l]):
                lambdas.append(self.lambda_history[l])
                Ls.append(self.L_history[l])
                losses.append(self.loss_history[l])
        return lambdas, Ls, losses
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
