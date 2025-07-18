# Levenberg-Marquardt algorithm
from typing import Sequence

import torch
import numpy as np

from .base import BaseOptimizer
from .. import AP_config
from . import func
from ..errors import OptimizeStopFail, OptimizeStopSuccess
from ..param import ValidContext

__all__ = ("LM",)


class LM(BaseOptimizer):
    """The LM class is an implementation of the Levenberg-Marquardt
    optimization algorithm. This method is used to solve non-linear
    least squares problems and is known for its robustness and
    efficiency.

    The Levenberg-Marquardt (LM) algorithm is an iterative method used
    for solving non-linear least squares problems. It can be seen as a
    combination of the Gauss-Newton method and the gradient descent
    method: it works like the Gauss-Newton method when the parameters
    are approximately near a minimum, and like the gradient descent method
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
    by Henri Gavin on which much of the AstroPhot LM implementation is
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

    The LM class implemented for AstroPhot takes a model, initial
    state, and various other optional parameters as inputs and seeks
    to find the parameters that minimize the cost function.

    Args:
      model: The model to be optimized.
      initial_state (Sequence): Initial values for the parameters to be optimized.
      max_iter (int): Maximum number of iterations for the algorithm.
      relative_tolerance (float): Tolerance level for relative change in cost function value to trigger termination of the algorithm.
      fit_parameters_identity: Used to select a subset of parameters. This is mostly used internally.
      verbose: Controls the verbosity of the output during optimization. A higher value results in more detailed output. If not provided, defaults to 0 (no output).
      max_step_iter (optional): The maximum number of steps while searching for chi^2 improvement on a single Jacobian evaluation. Default is 10.
      curvature_limit (optional): Controls how cautious the optimizer is for changing curvature. It should be a number greater than 0, where smaller is more cautious. Default is 1.
      Lup and Ldn (optional): These adjust the step sizes for the damping parameter. Default is 5 and 3 respectively.
      L0 (optional): This is the starting damping parameter. For easy problems with good initialization, this can be set lower. Default is 1.
      acceleration (optional): Controls the use of geodesic acceleration, which can be helpful in some scenarios. Set 1 for full acceleration, 0 for no acceleration. Default is 0.

    Here is some basic usage of the LM optimizer:

    .. code-block:: python

      import astrophot as ap

      # build model
      # ...

      # Initialize model parameters
      model.initialize()

      # Fit the parameters
      result = ap.fit.lm(model, verbose=1)

      # Check that a minimum was found
      print(result.message)

      # See the minimum chi^2 value
      print(f"min chi2: {result.res_loss()}")

      # Update parameter uncertainties
      result.update_uncertainty()

      # Extract multivariate Gaussian of uncertainties
      mu = result.res()
      cov = result.covariance_matrix

    """

    def __init__(
        self,
        model,
        initial_state: Sequence = None,
        max_iter: int = 100,
        relative_tolerance: float = 1e-5,
        Lup=11.0,
        Ldn=9.0,
        L0=1.0,
        max_step_iter: int = 10,
        ndf=None,
        **kwargs,
    ):

        super().__init__(
            model,
            initial_state,
            max_iter=max_iter,
            relative_tolerance=relative_tolerance,
            **kwargs,
        )
        # Maximum number of iterations of the algorithm
        self.max_iter = max_iter
        # Maximum number of steps while searching for chi^2 improvement on a single jacobian evaluation
        self.max_step_iter = max_step_iter
        # These are the adjustment step sized for the damping parameter
        self.Lup = Lup
        self.Ldn = Ldn
        # This is the starting damping parameter, for easy problems with good initialization, this can be set lower
        self.L = L0
        # mask
        fit_mask = self.model.fit_mask()
        if isinstance(fit_mask, tuple):
            fit_mask = torch.cat(tuple(FM.flatten() for FM in fit_mask))
        else:
            fit_mask = fit_mask.flatten()
        if torch.sum(fit_mask).item() == 0:
            fit_mask = None

        if model.target.has_mask:
            mask = self.model.target[self.fit_window].flatten("mask")
            if fit_mask is not None:
                mask = mask | fit_mask
            self.mask = ~mask
        elif fit_mask is not None:
            self.mask = ~fit_mask
        else:
            self.mask = torch.ones_like(
                self.model.target[self.fit_window].flatten("data"), dtype=torch.bool
            )
        if self.mask is not None and torch.sum(self.mask).item() == 0:
            raise OptimizeStopSuccess("No data to fit. All pixels are masked")

        # Initialize optimizer attributes
        self.Y = self.model.target[self.fit_window].flatten("data")[self.mask]

        # 1 / (sigma^2)
        kW = kwargs.get("W", None)
        if kW is not None:
            self.W = torch.as_tensor(
                kW, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            ).flatten()[self.mask]
        elif model.target.has_variance:
            self.W = self.model.target[self.fit_window].flatten("weight")[self.mask]
        else:
            self.W = torch.ones_like(self.Y)

        # The forward model which computes the output image given input parameters
        self.forward = lambda x: model(window=self.fit_window, params=x).flatten("data")[self.mask]
        # Compute the jacobian in representation units (defined for -inf, inf)
        self.jacobian = lambda x: model.jacobian(window=self.fit_window, params=x).flatten("data")[
            self.mask
        ]

        # variable to store covariance matrix if it is ever computed
        self._covariance_matrix = None

        # Degrees of freedom
        if ndf is None:
            self.ndf = max(1.0, len(self.Y) - len(self.current_state))
        else:
            self.ndf = ndf

    def chi2_ndf(self):
        return torch.sum(self.W * (self.Y - self.forward(self.current_state)) ** 2) / self.ndf

    @torch.no_grad()
    def fit(self) -> BaseOptimizer:
        """This performs the fitting operation. It iterates the LM step
        function until convergence is reached. Includes a message
        after fitting to indicate how the fitting exited. Typically if
        the message returns a "success" then the algorithm found a
        minimum. This may be the desired solution, or a pathological
        local minimum, this often depends on the initial conditions.

        """

        if len(self.current_state) == 0:
            if self.verbose > 0:
                AP_config.ap_logger.warning("No parameters to optimize. Exiting fit")
            self.message = "No parameters to optimize. Exiting fit"
            return self

        self._covariance_matrix = None
        self.loss_history = [self.chi2_ndf().item()]
        self.L_history = [self.L]
        self.lambda_history = [self.current_state.detach().clone().cpu().numpy()]
        if self.verbose > 0:
            AP_config.ap_logger.info(
                f"==Starting LM fit for '{self.model.name}' with {len(self.current_state)} dynamic parameters and {len(self.Y)} pixels=="
            )

        for _ in range(self.max_iter):
            if self.verbose > 0:
                AP_config.ap_logger.info(f"Chi^2/DoF: {self.loss_history[-1]:.6g}, L: {self.L:.3g}")
            try:
                if self.fit_valid:
                    with ValidContext(self.model):
                        res = func.lm_step(
                            x=self.model.to_valid(self.current_state),
                            data=self.Y,
                            model=self.forward,
                            weight=self.W,
                            jacobian=self.jacobian,
                            ndf=self.ndf,
                            chi2=self.loss_history[-1],
                            L=self.L,
                            Lup=self.Lup,
                            Ldn=self.Ldn,
                        )
                    self.current_state = self.model.from_valid(res["x"]).detach()
                else:
                    res = func.lm_step(
                        x=self.current_state,
                        data=self.Y,
                        model=self.forward,
                        weight=self.W,
                        jacobian=self.jacobian,
                        ndf=self.ndf,
                        chi2=self.loss_history[-1],
                        L=self.L,
                        Lup=self.Lup,
                        Ldn=self.Ldn,
                    )
                    self.current_state = res["x"].detach()
            except OptimizeStopFail:
                if self.verbose > 0:
                    AP_config.ap_logger.warning("Could not find step to improve Chi^2, stopping")
                self.message = self.message + "fail. Could not find step to improve Chi^2"
                break
            except OptimizeStopSuccess as e:
                if self.verbose > 0:
                    AP_config.ap_logger.info(f"Optimization converged successfully: {e}")
                self.message = self.message + "success"
                break

            self.L = np.clip(res["L"], 1e-9, 1e9)
            self.L_history.append(res["L"])
            self.loss_history.append(res["chi2"])
            self.lambda_history.append(self.current_state.detach().clone().cpu().numpy())

            if len(self.loss_history) >= 3:
                if (self.loss_history[-3] - self.loss_history[-1]) / self.loss_history[
                    -1
                ] < self.relative_tolerance and self.L < 0.1:
                    self.message = self.message + "success"
                    break
            if len(self.loss_history) > 10:
                if (self.loss_history[-10] - self.loss_history[-1]) / self.loss_history[
                    -1
                ] < self.relative_tolerance:
                    self.message = (
                        self.message + "success by immobility. Convergence not guaranteed"
                    )
                    break

        else:
            self.message = self.message + "fail. Maximum iterations"

        if self.verbose > 0:
            AP_config.ap_logger.info(
                f"Final Chi^2/DoF: {self.loss_history[-1]:.6g}, L: {self.L_history[-1]:.3g}. Converged: {self.message}"
            )

        self.model.fill_dynamic_values(self.current_state)

        return self

    @property
    @torch.no_grad()
    def covariance_matrix(self) -> torch.Tensor:
        """The covariance matrix for the model at the current
        parameters. This can be used to construct a full Gaussian PDF
        for the parameters using: :math:`\\mathcal{N}(\\mu,\\Sigma)`
        where :math:`\\mu` is the optimized parameters and
        :math:`\\Sigma` is the covariance matrix.

        """

        if self._covariance_matrix is not None:
            return self._covariance_matrix
        J = self.jacobian(self.current_state)
        hess = func.hessian(J, self.W)
        try:
            self._covariance_matrix = torch.linalg.inv(hess)
        except:
            AP_config.ap_logger.warning(
                "WARNING: Hessian is singular, likely at least one parameter is non-physical. Will massage Hessian to continue but results should be inspected."
            )
            hess += torch.eye(len(hess), dtype=AP_config.ap_dtype, device=AP_config.ap_device) * (
                torch.diag(hess) == 0
            )
            self._covariance_matrix = torch.linalg.inv(hess)
        return self._covariance_matrix

    @torch.no_grad()
    def update_uncertainty(self) -> None:
        """Call this function after optimization to set the uncertainties for
        the parameters. This will use the diagonal of the covariance
        matrix to update the uncertainties. See the covariance_matrix
        function for the full representation of the uncertainties.

        """
        # set the uncertainty for each parameter
        cov = self.covariance_matrix
        if torch.all(torch.isfinite(cov)):
            try:
                self.model.fill_dynamic_value_uncertainties(torch.sqrt(torch.abs(torch.diag(cov))))
            except RuntimeError as e:
                AP_config.ap_logger.warning(f"Unable to update uncertainty due to: {e}")
        else:
            AP_config.ap_logger.warning(
                "Unable to update uncertainty due to non finite covariance matrix"
            )
