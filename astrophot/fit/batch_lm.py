from ..models import Model
from ..image import TargetImageBatch, WindowBatch
from .base import BaseOptimizer
from ..backend_obj import backend
from .. import config
from ..errors import OptimizeStopFail, OptimizeStopSuccess


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
        max_step_iter: int = 10,
        likelihood="gaussian",
        constraint: Optional[LMConstraint] = None,
        **kwargs,
    ):

        super().__init__(
            model=model,
            initial_state=model.get_values(),
            max_iter=max_iter,
            relative_tolerance=relative_tolerance,
            **kwargs,
        )

        self.Lup = Lup
        self.Ldn = Ldn
        self.L = L0

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
        self.ndf = backend.sum(self.mask, axis=1) - self.current_state.shape[1]

    def chi2_ndf(self):
        return (
            backend.sum(
                self.weight * self.mask * (self.data - self.forward(self.current_state)) ** 2,
                axis=1,
            )
            / self.ndf
        )

    def poisson_2nll_ndf(self):
        M = self.forward(self.current_state)
        return (
            2 * backend.sum((M - self.data * backend.log(M + 1e-10)) * self.mask, axis=1) / self.ndf
        )

    def fit(self, update_uncertainty=True):
        if self.current_state.shape[1] == 0:
            if self.verbose > 0:
                config.logger.warning("No parameters to optimize. Exiting fit")
            self.message = "No parameters to optimize. Exiting fit"
            return self
