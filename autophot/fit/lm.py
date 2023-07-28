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
    """The LM class is an implementation of the Levenberg-Marquardt
    optimization algorithm. This method is used to solve non-linear
    least squares problems and is known for its robustness and
    efficiency.

    The Levenberg-Marquardt (LM) algorithm is an iterative method used
    for solving non-linear least squares problems. It can be seen as a
    combination of the Gauss-Newton method and the gradient descent
    method: it works like the Gauss-Newton method when the parameters
    are approximately correct, and like the gradient descent method
    when the parameters are far from their optimal values.

    The cost function that the LM algorithm tries to minimize is of
    the form:
    
    .. math::
        f(\\boldsymbol{\\beta}) = \\frac{1}{2}\\sum_{i=1}^{N} r_i(\\boldsymbol{\\beta})^2

    where :math:`\\boldsymbol{\\beta}` is the vector of parameters,
    :math:`r_i` are the residuals, and :math:`N` is the number of
    observations.

    The LM algorithm iteratively performs the following update to the parameters:

    .. math::
        \\boldsymbol{\\beta}_{n+1} = \\boldsymbol{\\beta}_{n} - (J^T J + \\lambda diag(J^T J))^{-1} J^T \\boldsymbol{r}

    where:
        - :math:`J` is the Jacobian matrix whose elements are :math:`J_{ij} = \\frac{\\partial r_i}{\\partial \\beta_j}`,
        - :math:`\\boldsymbol{r}` is the vector of residuals :math:`r_i(\\boldsymbol{\\beta})`,
        - :math:`\\lambda` is a damping factor which is adjusted at each iteration. 

    When :math:`\\lambda = 0` this can be seen as the Gauss-Newton
    method. In the limit that :math:`\\lambda` is large, the
    :math:`J^T J` matrix (an approximation of the Hessian) becomes
    subdominant and the update essentially points along :math:`J^T
    \\boldsymbol{r}` which is the gradient. In this scenario the
    gradient descent direction is also modified by the :math:`\\lambda
    diag(J^T J)` scaling which in some sense makes each gradient
    unitless and further improves the step. Note as well that as
    :math:`\\lambda` gets larger the step taken will be smaller, which
    helps to ensure convergence when the initial guess of the
    parameters are far from the optimal solution.

    Note that the residuals :math:`r_i` are typically also scaled by
    the variance of the pixels, but this does not change the equations
    above. For a detailed explanation of the LM method see the article
    by Henri Gavin on which much of the AutoPhot LM implementation is
    based::
    
        @article{Gavin2019,
            title={The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems},
            author={Gavin, Henri P},
            journal={Department of Civil and Environmental Engineering, Duke University},
            volume={19},
            year={2019}
        }

    as well as the paper on LM geodesic acceleration by Mark
    Transtrum::
    
        @article{Tanstrum2012,
           author = {{Transtrum}, Mark K. and {Sethna}, James P.},
            title = "{Improvements to the Levenberg-Marquardt algorithm for nonlinear least-squares minimization}",
             year = 2012,
              doi = {10.48550/arXiv.1201.5885},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2012arXiv1201.5885T},
        }

    The damping factor :math:`\\lambda` is adjusted at each iteration:
    it is effectively increased when we are far from the solution, and
    decreased when we are close to it. In practice, the algorithm
    attempts to pick the smallest :math:`\\lambda` that is can while
    making sure that the :math:`\\chi^2` decreases at each step.

    The main advantage of the LM algorithm is its adaptability. When
    the current estimate is far from the optimum, the algorithm
    behaves like a gradient descent to ensure global
    convergence. However, when it is close to the optimum, it behaves
    like Gauss-Newton, which provides fast local convergence.

    In practice, the algorithm is typically implemented with various
    enhancements to improve its performance. For example, the Jacobian
    may be approximated with finite differences, geodesic acceleration
    can be used to speed up convergence, and more sophisticated
    strategies can be used to adjust the damping factor :math:`\\lambda`.

    The exact performance of the LM algorithm will depend on the
    nature of the problem, including the complexity of the function
    f(x), the quality of the initial estimate x0, and the distribution
    of the data.

    The LM class implemented for AutoPhot takes a model, initial
    state, and various other optional parameters as inputs and seeks
    to find the parameters that minimize the cost function.

    Args:
      model: The model to be optimized.
      initial_state (Sequence): Initial values for the parameters to be optimized.
      max_iter (int): Maximum number of iterations for the algorithm.
      relative_tolerance (float): Tolerance level for relative change in cost function value to trigger termination of the algorithm.
      fit_parameters_identity: Used to select a subset of parameters .
      verbose: Controls the verbosity of the output during optimization. A higher value results in more detailed output. If not provided, defaults to 0 (no output).
      max_step_iter (optional): The maximum number of steps while searching for chi^2 improvement on a single Jacobian evaluation. Default is 10.
      curvature_limit (optional): Controls how cautious the optimizer is for changing curvature. It should be a number greater than 0, where smaller is more cautious. Default is 0.75.
      Lup and Ldn (optional): These are the adjustment step sizes for the damping parameter. Default is 5 and 3 respectively.
      L0 (optional): This is the starting damping parameter. For easy problems with good initialization, this can be set lower. Default is 1.
      acceleration (optional): Controls the use of geodesic acceleration, which can be helpful in some scenarios. Set 1 for full acceleration, 0 for no acceleration. Default is 0.
        
    """

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
        # The forward model which computes the output image given input parameters
        self.forward = partial(model, as_representation = True, parameters_identity = fit_parameters_identity)
        # Compute the jacobian in representation units (defined for -inf, inf)
        self.jacobian = partial(model.jacobian, as_representation = True, parameters_identity = fit_parameters_identity)
        # Compute the jacobian in natural units
        self.jacobian_natural = partial(model.jacobian, as_representation = False, parameters_identity = fit_parameters_identity)
        # Function to transform between true parameter values and representation values
        self.transform = partial(model.parameters.transform, to_representation = False, parameters_identity = fit_parameters_identity)
        # Maximum number of iterations of the algorithm
        self.max_iter = max_iter
        # Maximum number of steps while searching for chi^2 improvement on a single jacobian evaluation
        self.max_step_iter = kwargs.get("max_step_iter", 10)
        # sets how cautious the optimizer is for changing curvature, should be number greater than 0, where smaller is more cautious
        self.curvature_limit = kwargs.get("curvature_limit", 0.75) 
        # These are the adjustment step sized for the damping parameter
        self._Lup = kwargs.get("Lup", 5.)
        self._Ldn = kwargs.get("Ldn", 3.)
        # This is the starting damping parameter, for easy problems with good initialization, this can be set lower
        self.L = kwargs.get("L0", 1.)
        # Geodesic acceleration is helpful in some scenarios. By default it is turned off. Set 1 for full acceleration, 0 for no acceleration.
        self.acceleration = kwargs.get("acceleration", 0.)
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

        # variable to store covariance matrix if it is ever computed
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
        scarry_best = (None, init_chi2, self.L)
        direction = "none"
        iteration = 0
        d = 0.1
        for iteration in range(self.max_step_iter):
            # In a scenario where LM is having a hard time proposing a good step, but the damping is really low, just jump up to normal damping levels
            if iteration > self.max_step_iter/2 and self.L < 1e-3:
                self.L = 1.

            # compute LM update step
            h = self._h(self.L, self.grad, self.hess)

            # Compute goedesic acceleration
            Y1 = self.forward(parameters = self.current_state + d*h).flatten("data")
            rh = -self.W * (self.Y - Y1)
            rpp = (2 / d) * ((rh - r) / d - self.W*(J @ h))
            if self.L > 1e-4:
                a = -self._h(self.L, J.T @ rpp, self.hess) / 2
            else:
                a = torch.zeros_like(h)

            # Evaluate new step
            ha = h + a*self.acceleration
            Y1 = self.forward(parameters = self.current_state + ha).flatten("data")

            # Compute and report chi^2
            chi2 = self._chi2(Y1.detach()).item()
            if self.verbose > 1:
                AP_config.ap_logger.info(f"sub step L: {self.L}, Chi^2: {chi2}")

            # Skip if chi^2 is nan
            if not np.isfinite(chi2):
                if self.verbose > 1:
                    AP_config.ap_logger.info("Skip due to non-finite values")
                self.Lup()
                if direction == "better":
                    break
                direction = "worse"
                continue
            
            # Keep track of chi^2 improvement even if it fails curvature test
            if chi2 <= scarry_best[1]:
                scarry_best = (ha, chi2, self.L)
                nostep = False

            # Check for high curvature, in which case linear approximation is not valid. avoid this step
            if torch.linalg.norm(a) / torch.linalg.norm(h) > self.curvature_limit:
                if self.verbose > 1:
                    AP_config.ap_logger.info("Skip due to large curvature")
                self.Lup()
                if direction == "better":
                    break
                direction = "worse"
                continue

            # Check for Chi^2 improvement
            if chi2 <= best[1]:
                if self.verbose > 1:
                    AP_config.ap_logger.info("new best chi^2")
                best = (ha, chi2, self.L)
                nostep = False
                self.Ldn()
                if self.L <= 1e-8 or direction == "worse":
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

            # If a step substantially improves the chi^2, stop searching for better step, simply exit the loop and accept the good step
            if (best[1] - init_chi2) / init_chi2 < -0.1:
                if self.verbose > 1:
                    AP_config.ap_logger.info("Large step taken, ending search for good step")
                break

        if nostep:
            if scarry_best[0] is not None:
                if self.verbose > 1:
                    AP_config.ap_logger.warn("no low curvature step found, taking high curvature step")
                return scarry_best
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

        self._covariance_matrix = None
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
