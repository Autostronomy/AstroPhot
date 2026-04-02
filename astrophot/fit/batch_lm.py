import numpy as np
from ..models import Model
from ..image import TargetImageBatch, WindowBatch
from .base import BaseOptimizer
from ..backend_obj import backend, ArrayLike
from .. import config
from ..errors import OptimizeStopSuccess
from ..param import ValidContext
from . import func


class BatchLM(BaseOptimizer):

    def __init__(
        self,
        model: Model,
        batch_target: TargetImageBatch,
        batch_window: WindowBatch,
        max_iter: int = 100,
        relative_tolerance: float = 1e-5,
        Lup=11.0,
        Ldn=9.0,
        L0=1.0,
        max_step_iter: int = 3,
        likelihood="gaussian",
        **kwargs,
    ):

        super().__init__(
            model=model,
            initial_state=model.get_values(),
            max_iter=max_iter,
            relative_tolerance=relative_tolerance,
            **kwargs,
        )

        self.max_step_iter = max_step_iter

        # Likelihood
        self.likelihood = likelihood
        if self.likelihood not in ["gaussian", "poisson"]:
            raise ValueError(
                f"Unsupported likelihood: {self.likelihood}, should be one of: 'gaussian' or 'poisson'"
            )

        # mask
        mask = backend.flatten(batch_target[batch_window].mask, 1, -1)
        self.mask = ~mask
        if backend.sum(self.mask).item() == 0:
            raise OptimizeStopSuccess("No data to fit. All pixels are masked")

        # data
        self.data = backend.flatten(batch_target[batch_window].data, 1, -1)

        # Weight
        self.weight = backend.flatten(batch_target[batch_window].weight, 1, -1)

        # WCS
        crtan = batch_target.crtan
        shift = backend.as_array(
            batch_window.origin_shifter(self.model.window), dtype=config.DTYPE, device=config.DEVICE
        )
        crpix = batch_target[batch_window].crpix + shift
        CD = batch_target.CD
        psf = batch_target.psf_stack
        psf_batch = None if psf is None else 0

        # Forward
        vmodel = backend.vmap(
            lambda cd, crt, crp, psf, params: backend.flatten(
                self.model(cd, crt, crp, psf, params=params).data
            ),
            in_dims=(0, 0, 0, psf_batch, 0),
        )
        self.forward = lambda x: vmodel(CD, crtan, crpix, psf, x)

        # Jacobian
        vjac = backend.vmap(
            backend.jacfwd(
                lambda cd, crt, crp, psf, params: backend.flatten(
                    self.model(cd, crt, crp, psf, params=params).data
                ),
                argnums=4,
            ),
            in_dims=(0, 0, 0, psf_batch, 0),
        )
        self.jacobian = lambda x: vjac(CD, crtan, crpix, psf, x)

        # ndf
        self.ndf = backend.clamp(
            backend.sum(self.mask, dim=1) - self.current_state.shape[1], backend.as_array(1), None
        )

        # LM parameters
        self.Lup = Lup
        self.Ldn = Ldn
        self.L = L0 * backend.ones(
            self.current_state.shape[0], dtype=config.DTYPE, device=config.DEVICE
        )

    def chi2_ndf(self):
        return (
            backend.sum(
                self.weight * self.mask * (self.data - self.forward(self.current_state)) ** 2,
                dim=1,
            )
            / self.ndf
        )

    def poisson_2nll_ndf(self):
        M = self.forward(self.current_state)
        return (
            2 * backend.sum((M - self.data * backend.log(M + 1e-10)) * self.mask, dim=1) / self.ndf
        )

    def fit(self, update_uncertainty=True):
        if self.current_state.shape[1] == 0:
            if self.verbose > 0:
                config.logger.warning("No parameters to optimize. Exiting fit")
            self.message = "No parameters to optimize. Exiting fit"
            return self

        if self.likelihood == "gaussian":
            quantity = "Chi^2/DoF"
            self.loss_history = [backend.to_numpy(self.chi2_ndf())]
        elif self.likelihood == "poisson":
            quantity = "2NLL/DoF"
            self.loss_history = [backend.to_numpy(self.poisson_2nll_ndf())]
        self._covariance_matrix = None
        self.L_history = [backend.to_numpy(self.L)]
        self.lambda_history = [backend.to_numpy(backend.copy(self.current_state))]
        if self.verbose > 0:
            config.logger.info(
                f"==Starting LM fit for '{self.model.name}' with batch of {self.current_state.shape[0]} images with {self.current_state.shape[1]} dynamic parameters and {self.data.shape[1]} pixels=="
            )

        for _ in range(self.max_iter):
            if self.verbose > 0:
                config.logger.info(f"{quantity}: {self.loss_history[-1]}, L: {self.L_history[-1]}")

            if self.fit_valid:
                with ValidContext(self.model):
                    res = func.batch_lm_step(
                        x=self.model.to_valid(self.current_state),
                        data=self.data,
                        model=self.forward,
                        weight=self.weight,
                        mask=self.mask,
                        jacobian=self.jacobian,
                        L=self.L,
                        Lup=self.Lup,
                        Ldn=self.Ldn,
                        likelihood=self.likelihood,
                        max_step_iter=self.max_step_iter,
                    )
                self.current_state = self.model.from_valid(backend.copy(res["x"]))
            else:
                res = func.batch_lm_step(
                    x=self.current_state,
                    data=self.data,
                    model=self.forward,
                    weight=self.weight,
                    mask=self.mask,
                    jacobian=self.jacobian,
                    L=self.L,
                    Lup=self.Lup,
                    Ldn=self.Ldn,
                    likelihood=self.likelihood,
                    max_step_iter=self.max_step_iter,
                )
                self.current_state = backend.copy(res["x"])

            self.L = backend.clamp(res["L"], backend.as_array(1e-9), backend.as_array(1e9))
            self.L_history.append(backend.to_numpy(self.L))
            self.loss_history.append(2 * res["nll"] / backend.to_numpy(self.ndf))
            self.lambda_history.append(backend.to_numpy(backend.copy(self.current_state)))

            if self.check_convergence():
                break
        else:
            self.message = self.message + "fail. Maximum iterations"

        if self.verbose > 0:
            config.logger.info(
                f"Final {quantity}: {self.loss_history[-1]}, L: {self.L_history[-1]}. Converged: {self.message}"
            )

        self.model.set_values(self.current_state)
        if update_uncertainty:
            self.update_uncertainty()

        return self

    def check_convergence(self) -> bool:
        """Check if the optimization has converged based on the last
        iteration's chi^2 and the relative tolerance.
        """
        if len(self.loss_history) < 3:
            return False
        if np.all(
            (self.loss_history[-2] - self.loss_history[-1]) / self.loss_history[-1]
            < self.relative_tolerance
        ) and np.all(backend.to_numpy(self.L) < 0.1):
            self.message = self.message + "success"
            return True
        if len(self.loss_history) < 10:
            return False
        if np.all(
            (self.loss_history[-10] - self.loss_history[-1]) / self.loss_history[-1]
            < self.relative_tolerance
        ):
            self.message = self.message + "success by immobility. Convergence not guaranteed"
            return True
        return False

    @property
    def covariance_matrix(self) -> ArrayLike:
        """The covariance matrix for the model at the current
        parameters. This can be used to construct a full Gaussian PDF for the
        parameters using: $\\mathcal{N}(\\mu,\\Sigma)$ where $\\mu$ is the
        optimized parameters and $\\Sigma$ is the covariance matrix.

        """

        if self._covariance_matrix is not None:
            return self._covariance_matrix
        J = self.jacobian(self.current_state) * self.mask.reshape(self.mask.shape + (1,))
        if self.likelihood == "gaussian":
            hess = backend.vmap(func.hessian)(J, self.weight * self.mask)
        elif self.likelihood == "poisson":
            hess = backend.vmap(func.hessian_poisson)(
                J, self.data * self.mask, self.forward(self.current_state) * self.mask
            )
        try:
            self._covariance_matrix = backend.vmap(backend.linalg.inv)(hess)
        except:
            config.logger.warning(
                "WARNING: Hessian is singular, likely at least one parameter is non-physical. Will use pseudo-inverse of Hessian to continue but results should be inspected."
            )
            self._covariance_matrix = backend.vmap(backend.linalg.pinv)(hess)
        return self._covariance_matrix

    def update_uncertainty(self) -> None:
        """Call this function after optimization to set the uncertainties for
        the parameters. This will use the diagonal of the covariance
        matrix to update the uncertainties. See the covariance_matrix
        function for the full representation of the uncertainties.

        """
        # set the uncertainty for each parameter
        cov = self.covariance_matrix
        if backend.all(backend.isfinite(cov)):
            try:
                self.model.set_values(
                    backend.sqrt(backend.abs(backend.vmap(backend.diag)(cov))),
                    attribute="uncertainty",
                )
            except RuntimeError as e:
                config.logger.warning(f"Unable to update uncertainty due to: {e}")
        else:
            config.logger.warning(
                "Unable to update uncertainty due to non finite covariance matrix"
            )
