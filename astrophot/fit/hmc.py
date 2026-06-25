# Hamiltonian Monte-Carlo
from typing import Optional, Sequence

try:
    import pyro
    import pyro.distributions as dist
    from pyro.distributions import Distribution
    from pyro.infer import MCMC as pyro_MCMC
    from pyro.infer import HMC as pyro_HMC
    from pyro.infer.mcmc.adaptation import BlockMassMatrix
    from pyro.ops.welford import WelfordCovariance
except ImportError:
    pyro = None
    Distribution = None

from .base import BaseOptimizer
from ..models import Model
from .. import config
from ..backend_obj import backend, ArrayLike

__all__ = ("HMC",)


###########################################
# !Overwrite pyro configuration behavior!
# currently this is the only way to provide
# mass matrix manually
###########################################
def new_configure(self, mass_matrix_shape, adapt_mass_matrix=True, options={}):
    """
    Sets up an initial mass matrix.

    :param mass_matrix_shape: a dict that maps tuples of site names to the shape of
        the corresponding mass matrix. Each tuple of site names corresponds to a block.
    :param adapt_mass_matrix: a flag to decide whether an adaptation scheme will be used.
    :param options: Array options to construct the initial mass matrix.
    """
    inverse_mass_matrix = {}
    for site_names, shape in mass_matrix_shape.items():
        self._mass_matrix_size[site_names] = shape[0]
        diagonal = len(shape) == 1
        inverse_mass_matrix[site_names] = (
            backend.ones(shape, **options) * self._init_scale
            if diagonal
            else backend.eye(shape[0], **options) * self._init_scale
        )
        if adapt_mass_matrix:
            adapt_scheme = WelfordCovariance(diagonal=diagonal)
            self._adapt_scheme[site_names] = adapt_scheme

    if len(self.inverse_mass_matrix.keys()) == 0:
        self.inverse_mass_matrix = inverse_mass_matrix


if pyro is not None:
    BlockMassMatrix.configure = new_configure
############################################


class HMC(BaseOptimizer):
    """Hamiltonian Monte-Carlo sampler wrapper for the Pyro package.

    This MCMC algorithm uses gradients of the $\\chi^2$ to more
    efficiently explore the probability distribution.

    More information on HMC can be found at:
    https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo,
    https://arxiv.org/abs/1701.02434, and
    http://www.mcmchandbook.net/HandbookChapter5.pdf

    :param max_iter: The number of sampling steps to perform. Defaults to 1000.
    :type max_iter: int, optional
    :param epsilon: The length of the integration step to perform for each leapfrog iteration. The momentum update will be of order epsilon * score. Defaults to 1e-5.
    :type epsilon: float, optional
    :param leapfrog_steps: Number of steps to perform with leapfrog integrator per sample of the HMC. Defaults to 10.
    :type leapfrog_steps: int, optional
    :param inv_mass: Inverse Mass matrix (covariance matrix) which can tune the behavior in each dimension to ensure better mixing when sampling. Defaults to the identity.
    :type inv_mass: float or array, optional
    :param progress_bar: Whether to display a progress bar during sampling. Defaults to True.
    :type progress_bar: bool, optional
    :param prior: Prior distribution for the parameters. Defaults to None.
    :type prior: distribution, optional
    :param warmup: Number of warmup steps before actual sampling begins. Defaults to 100.
    :type warmup: int, optional
    :param hmc_kwargs: Additional keyword arguments for the HMC sampler. Defaults to {}.
    :type hmc_kwargs: dict, optional
    :param mcmc_kwargs: Additional keyword arguments for the MCMC process. Defaults to {}.
    :type mcmc_kwargs: dict, optional

    """

    def __init__(
        self,
        model: Model,
        initial_state: Optional[Sequence] = None,
        max_iter: int = 1000,
        inv_mass: Optional[ArrayLike] = None,
        epsilon: float = 1e-4,
        leapfrog_steps: int = 10,
        progress_bar: bool = True,
        prior: Optional["Distribution"] = None,
        warmup: int = 100,
        hmc_kwargs: dict = {},
        mcmc_kwargs: dict = {},
        likelihood: str = "gaussian",
        **kwargs,
    ):
        if pyro is None:
            raise ImportError("Pyro must be installed to use HMC.")
        super().__init__(model, initial_state, max_iter=max_iter, **kwargs)

        self.inv_mass = inv_mass
        self.epsilon = epsilon
        self.leapfrog_steps = leapfrog_steps
        self.progress_bar = progress_bar
        self.prior = prior
        self.warmup = warmup
        self.hmc_kwargs = hmc_kwargs
        self.mcmc_kwargs = mcmc_kwargs
        self.likelihood = likelihood
        self.acceptance = None

    def fit(
        self,
        state: Optional[ArrayLike] = None,
    ):
        """Performs MCMC sampling using Hamiltonian Monte-Carlo step.

        Records the chain for later examination.

        :param state: Model parameters as a 1D Array.
        :type state: Array, optional

        """

        def step(model, prior):
            x = pyro.sample("x", prior)
            # Log-likelihood function
            if self.likelihood == "gaussian":
                log_likelihood_value = model.gaussian_log_likelihood(params=x)
            elif self.likelihood == "poisson":
                log_likelihood_value = model.poisson_log_likelihood(params=x)
            else:
                raise ValueError(f"Unsupported likelihood type: {self.likelihood}")
            # Observe the log-likelihood
            pyro.factor("obs", log_likelihood_value)

        if self.prior is None:
            self.prior = dist.Normal(
                self.current_state,
                backend.ones_like(self.current_state) * 1e2 + backend.abs(self.current_state) * 1e2,
            )

        # Set up the HMC sampler
        hmc_kwargs = {
            "jit_compile": False,
            "ignore_jit_warnings": True,
            "full_mass": True,
            "step_size": self.epsilon,
            "num_steps": self.leapfrog_steps,
            "adapt_step_size": False,
            "adapt_mass_matrix": self.inv_mass is None,
        }
        hmc_kwargs.update(self.hmc_kwargs)
        hmc_kernel = pyro_HMC(step, **hmc_kwargs)
        if self.inv_mass is not None:
            hmc_kernel.mass_matrix_adapter.inverse_mass_matrix = {("x",): self.inv_mass}

        # Provide an initial guess for the parameters
        init_params = {"x": self.model.get_values()}

        # Run MCMC with the HMC sampler and the initial guess
        mcmc_kwargs = {
            "num_samples": self.max_iter,
            "warmup_steps": self.warmup,
            "initial_params": init_params,
            "disable_progbar": not self.progress_bar,
        }
        mcmc_kwargs.update(self.mcmc_kwargs)
        mcmc = pyro_MCMC(hmc_kernel, **mcmc_kwargs)

        mcmc.run(self.model, self.prior)
        self.iteration += self.max_iter

        # Extract posterior samples
        chain = mcmc.get_samples()["x"]

        self.chain = chain
        self.model.set_values(
            backend.as_array(self.chain[-1], dtype=config.DTYPE, device=config.DEVICE)
        )
        return self
