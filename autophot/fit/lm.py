# Levenberg-Marquardt algorithm
import os
from time import time
from typing import List, Callable, Optional, Union, Sequence, Any
from functools import partial

import torch
from torch.autograd.functional import jacobian
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseOptimizer
from .. import AP_config

__all__ = ("LM",)

class OptimizeStopFail(Exception):
    pass

class LM(BaseOptimizer):

    def __init__(
            self,
            model,
            initial_state: Sequence = None,
            max_iter: int = 100,
            relative_tolerance: float = 1e-5,
            fit_parameters_identity=None,
            **kwargs,
    ):

        super().__init__(
            model,
            initial_state,
            max_iter=max_iter,
            fit_parameters_identity=fit_parameters_identity,
            relative_tolerance = relative_tolerance,
            **kwargs,
        )
        self.forward = partial(model, as_representation = True, parameters_identity = fit_parameters_identity)
        self.jacobian = partial(model.jacobian, as_representation = True, parameters_identity = fit_parameters_identity)
        self.jacobian_natural = partial(model.jacobian, as_representation = False, parameters_identity = fit_parameters_identity)
        self.transform = partial(model.parameters.transform, to_representation = False, parameters_identity = fit_parameters_identity)
        self.max_iter = max_iter
        self.max_step_iter = kwargs.get("max_step_iter", 10)
        self.curvature_limit = kwargs.get("curvature_limit", 0.75) # sets how cautious the optimizer is for changing curvature, should be number greater than 0, where smaller is more cautious
        self._Lup = 3.
        self._Ldn = 2.
        self.L = kwargs.get("L0", 1.)
        # Initialize optimizer atributes
        self.Y = self.model.target[self.fit_window].flatten("data")
        
        # Degrees of freedom
        self.ndf = len(self.Y) - len(self.current_state)

        # 1 / (2 * sigma^2)
        if model.target.has_variance:
            self.W = 1. / self.model.target[self.fit_window].flatten("variance")
        else:
            self.W = torch.ones_like(self.Y)

        # mask
        if model.target.has_mask:
            mask = self.model.target[self.fit_window].flatten("mask")
            self.mask = torch.logical_not(mask)
            self.ndf -= torch.sum(mask).item()
        else:
            self.mask = None

        self._covariance_matrix = None

    def Lup(self):
        self.L = min(1e9, self.L * self._Lup)
    def Ldn(self):
        self.L = max(1e-9, self.L / self._Ldn)
        
    @torch.no_grad()
    def step(self, chi2):

        Y0 = self.forward(parameters = self.current_state).flatten("data")
        J = self.jacobian(parameters = self.current_state).flatten("data")
        r = -self.W * (self.Y - Y0)
        self.hess = J.T @ (self.W.view(len(self.W), -1) * J)
        self.grad = J.T @ (self.W * (self.Y - Y0))

        init_chi2 = chi2
        nostep = True
        best = (torch.zeros_like(self.current_state), init_chi2, self.L)
        direction = "none"
        iteration = 0
        d = 0.1
        for iteration in range(self.max_step_iter):
            h = self._h(self.L, self.grad, self.hess)
            Y1 = self.forward(parameters = self.current_state + d*h).flatten("data")
            rh = -self.W * (self.Y - Y1)
                
            rpp = (2 / d) * ((rh - r) / d - self.W*(J @ h))
            a = -self._h(self.L, J.T @ rpp, self.hess) / 2
            if 2 * torch.linalg.norm(a) / torch.linalg.norm(h) > self.curvature_limit:
                if self.verbose > 1:
                    AP_config.ap_logger.info("Skip due to large curvature")
                self.Lup()
                if direction == "better":
                    break
                direction = "worse"
                continue
            ha = h + a
            Y1 = self.forward(parameters = self.current_state + ha).flatten("data")

            chi2 = self._chi2(Y1.detach()).item()
            if self.verbose > 1:
                AP_config.ap_logger.info(f"sub step L: {self.L}, Chi^2: {chi2}")

            if not np.isfinite(chi2):
                if self.verbose > 1:
                    AP_config.ap_logger.info("Skip due to non-finite values")
                self.Lup()
                if direction == "better":
                    break
                direction = "worse"
                continue

            if chi2 <= best[1]:
                if self.verbose > 1:
                    AP_config.ap_logger.info("new best chi^2")
                best = (ha, chi2, self.L)
                nostep = False
                self.Ldn()
                if self.L == 1e-9 or direction == "worse":
                    break
                direction = "better"
            elif chi2 > best[1] and direction in ["none", "worse"]:
                if self.verbose > 1:
                    AP_config.ap_logger.info("chi^2 is worse")
                self.Lup()
                if self.L == 1e9:
                    break
                direction = "worse"
            else:
                break

            if (best[1] - init_chi2) / init_chi2 < -0.1:
                if self.verbose > 1:
                    AP_config.ap_logger.info("Large step taken, ending search for good step")
                break

        if nostep:
            raise OptimizeStopFail("Could not find step to improve chi^2")

        return best

    @staticmethod
    @torch.no_grad()
    def _h(L, grad, hess):

        I = torch.eye(len(grad), dtype=grad.dtype, device=grad.device)

        h = torch.linalg.solve(
            (hess + 1e-2 * L**2 * I) * (1 + L**2 * I) ** 2 / (1 + L**2),
            grad,
        )
        
        return h

    @torch.no_grad()
    def _chi2(self, Ypred):
        if self.mask is None:
            return torch.sum(self.W * (self.Y - Ypred)**2) / self.ndf
        else:
            return torch.sum((self.W * (self.Y - Ypred)**2)[self.mask]) / self.ndf
            

    @torch.no_grad()
    def update_hess_grad(self, natural = False):

        if natural:
            J = self.jacobian_natural(parameters = self.transform(self.current_state)).flatten("data")
        else:
            J = self.jacobian(parameters = self.current_state).flatten("data")
        Ypred = self.forward(parameters = self.current_state).flatten("data")
        self.hess = torch.matmul(J.T, (self.W.view(len(self.W), -1) * J))
        self.grad = torch.matmul(J.T, self.W * (self.Y - Ypred))
        
    @torch.no_grad()
    def fit(self):

        self.loss_history = [self._chi2(self.forward(parameters = self.current_state).flatten("data")).item()]
        self.L_history = [self.L]
        self.lambda_history = [self.current_state.detach().clone().cpu().numpy()]
        
        for iteration in range(self.max_iter):
            if self.verbose > 0:
                AP_config.ap_logger.info(f"Chi^2: {self.loss_history[-1]}, L: {self.L}")
            try:
                res = self.step(chi2 = self.loss_history[-1])
            except OptimizeStopFail:
                if self.verbose > 0:
                    AP_config.ap_logger.warning("Could not find step to improve Chi^2, stopping")
                self.message = self.message + "fail. Could not find step to improve Chi^2"
                break

            self.L = res[2]
            self.current_state = (self.current_state + res[0]).detach()
            
            self.L_history.append(self.L)
            self.loss_history.append(res[1])
            self.lambda_history.append(self.current_state.detach().clone().cpu().numpy())
            
            self.Ldn()
            
            if len(self.loss_history) >= 3:
                if (self.loss_history[-3] - self.loss_history[-1]) / self.loss_history[-1] < self.relative_tolerance and self.L < 0.1:
                    self.message = self.message + "success"
                    break
            if len(self.loss_history) > 10:
                if (self.loss_history[-10] - self.loss_history[-1]) / self.loss_history[-1] < self.relative_tolerance:
                    self.message = self.message + "success by immobility. Convergence not guaranteed"
                    break
                
        else:
            self.message = self.message + "fail. Maximum iterations"
                
        if self.verbose > 0:
            AP_config.ap_logger.info(f"Final Chi^2: {self.loss_history[-1]}, L: {self.L_history[-1]}. Converged: {self.message}")
        self.model.parameters.set_values(
            self.res(),
            as_representation=True,
            parameters_identity=self.fit_parameters_identity,
        )

        return self

    @property
    @torch.no_grad()
    def covariance_matrix(self) -> torch.Tensor:
        if self._covariance_matrix is not None:
            return self._covariance_matrix
        self.update_hess_grad(natural = True)
        try:
            self._covariance_matrix = torch.linalg.inv(self.hess)
        except:
            AP_config.ap_logger.warning(
                "WARNING: Hessian is singular, likely at least one model is non-physical. Will massage Hessian to continue but results should be inspected."
            )
            self.hess += torch.eye(
                len(self.grad), dtype=AP_config.ap_dtype, device=AP_config.ap_device
            ) * (torch.diag(self.hess) == 0)
            self._covariance_matrix = torch.linalg.inv(self.hess)
        return self._covariance_matrix

    @torch.no_grad()
    def update_uncertainty(self):
        # set the uncertainty for each parameter
        cov = self.covariance_matrix
        if torch.all(torch.isfinite(cov)):
            try:
                self.model.parameters.set_uncertainty(
                    torch.sqrt(
                        torch.abs(torch.diag(cov))
                    ),
                    as_representation=False,
                    parameters_identity=self.fit_parameters_identity,
                )
            except RuntimeError as e:
                AP_config.ap_logger.warning(f"Unable to update uncertainty due to: {e}")
        else:
            AP_config.ap_logger.warning(f"Unable to update uncertainty due to non finite covariance matrix")
