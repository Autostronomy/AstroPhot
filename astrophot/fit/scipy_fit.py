from typing import Sequence

import torch
from scipy.optimize import minimize

from .base import BaseOptimizer
from .. import AP_config
from ..errors import OptimizeStopSuccess

__all__ = ("ScipyFit",)


class ScipyFit(BaseOptimizer):

    def __init__(
        self,
        model,
        initial_state: Sequence = None,
        method="Nelder-Mead",
        max_iter: int = 100,
        ndf=None,
        **kwargs,
    ):

        super().__init__(
            model,
            initial_state,
            max_iter=max_iter,
            **kwargs,
        )
        self.method = method
        # Maximum number of iterations of the algorithm
        self.max_iter = max_iter
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

    def chi2_ndf(self, x):
        return torch.sum(self.W * (self.Y - self.forward(x)) ** 2) / self.ndf

    def numpy_bounds(self):
        """Convert the model's parameter bounds to a format suitable for scipy.optimize."""
        bounds = []
        for param in self.model.dynamic_params:
            if param.shape == ():
                bound = [None, None]
                if param.valid[0] is not None:
                    bound[0] = param.valid[0].detach().cpu().numpy()
                if param.valid[1] is not None:
                    bound[1] = param.valid[1].detach().cpu().numpy()
                bounds.append(tuple(bound))
            else:
                for i in range(param.value.numel()):
                    bound = [None, None]
                    if param.valid[0] is not None:
                        bound[0] = param.valid[0].flatten()[i].detach().cpu().numpy()
                    if param.valid[1] is not None:
                        bound[1] = param.valid[1].flatten()[i].detach().cpu().numpy()
                    bounds.append(tuple(bound))
        return bounds

    def fit(self):

        res = minimize(
            lambda x: self.chi2_ndf(
                torch.tensor(x, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
            ).item(),
            self.current_state,
            method=self.method,
            bounds=self.numpy_bounds(),
            options={
                "maxiter": self.max_iter,
            },
        )
        self.scipy_res = res
        self.message = self.message + f"success: {res.success}, message: {res.message}"
        self.current_state = torch.tensor(
            res.x, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        if self.verbose > 0:
            AP_config.ap_logger.info(
                f"Final Chi^2/DoF: {self.chi2_ndf(self.current_state):.6g}. Converged: {self.message}"
            )
        self.model.fill_dynamic_values(self.current_state)

        return self
