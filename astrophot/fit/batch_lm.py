from ..models import Model
from ..image import TargetImageBatch
from .base import BaseOptimizer


class BatchLM(BaseOptimizer):

    def __init__(
        self,
        model: Model,
        batch_target: TargetImageBatch,
        max_iter: int = 100,
        relative_tolerance: float = 1e-5,
        Lup=11.0,
        Ldn=9.0,
        L0=1.0,
        max_step_iter: int = 10,
        ndf=None,
        likelihood="gaussian",
        constraint: Optional[LMConstraint] = None,
        forward=None,
        jacobian=None,
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
