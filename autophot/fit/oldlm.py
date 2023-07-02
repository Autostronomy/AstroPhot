# Levenberg-Marquardt algorithm
import os
from time import time
from typing import List, Callable, Optional, Union, Sequence, Any

import torch
from torch.autograd.functional import jacobian
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseOptimizer
from .. import AP_config

__all__ = ["oldLM", "LM_Constraint"]


@torch.no_grad()
@torch.jit.script
def Broyden_step(J, h, Yp, Yph):
    delta = torch.matmul(J, h)
    # avoid constructing a second giant jacobian matrix, instead go one row at a time
    for j in range(J.shape[1]):
        J[:, j] += (Yph - Yp - delta) * h[j] / torch.linalg.norm(h)
    return J


class oldLM(BaseOptimizer):
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

    Args:
        model (AutoPhot_Model): object with which to perform optimization
        initial_state (Optional[Sequence]): an initial state for optimization
        epsilon4 (Optional[float]): approximation accuracy requirement, for any rho < epsilon4 the step will be rejected. Default 0.1
        epsilon5 (Optional[float]): numerical stability factor, added to the diagonal of the Hessian. Default 1e-8
        constraints (Optional[Union[LM_Constraint,tuple[LM_Constraint]]]): Constraint objects which control the fitting process.
        L0 (Optional[float]): initial value for L factor in (H +L*I)h = G. Default 1.
        Lup (Optional[float]): amount to increase L when rejecting an update step. Default 11.
        Ldn (Optional[float]): amount to decrease L when accetping an update step. Default 9.

    """

    def __init__(
        self,
        model: "AutoPhot_Model",
        initial_state: Sequence = None,
        max_iter: int = 100,
        fit_parameters_identity: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(
            model,
            initial_state,
            max_iter=max_iter,
            fit_parameters_identity=fit_parameters_identity,
            **kwargs,
        )

        # Set optimizer parameters
        self.epsilon4 = kwargs.get("epsilon4", 0.1)
        self.epsilon5 = kwargs.get("epsilon5", 1e-8)
        self.Lup = kwargs.get("Lup", 11.0)
        self.Ldn = kwargs.get("Ldn", 9.0)
        self.L = kwargs.get("L0", 1e-3)
        self.use_broyden = kwargs.get("use_broyden", False)

        # Initialize optimizer atributes
        self.Y = self.model.target[self.fit_window].flatten("data")
        #        1 / sigma^2
        self.W = (
            1.0 / self.model.target[self.fit_window].flatten("variance")
            if model.target.has_variance
            else 1.0
        )
        #          # pixels      # parameters
        self.ndf = len(self.Y) - len(self.current_state)
        self.J = None
        self.full_jac = False
        self.current_Y = None
        self.prev_Y = [None, None]
        if self.model.target.has_mask:
            self.mask = self.model.target[self.fit_window].flatten("mask")
            # subtract masked pixels from degrees of freedom
            self.ndf -= torch.sum(self.mask)
        self.L_history = []
        self.decision_history = []
        self.rho_history = []
        self._count_converged = 0
        self.ndf = kwargs.get("ndf", self.ndf)
        self._covariance_matrix = None

        # update attributes with constraints
        self.constraints = kwargs.get("constraints", None)
        if self.constraints is not None and isinstance(self.constraints, LM_Constraint):
            self.constraints = (self.constraints,)

        if self.constraints is not None:
            for con in self.constraints:
                self.Y = torch.cat((self.Y, con.reference_value))
                self.W = torch.cat((self.W, 1 / con.weight))
                self.ndf -= con.reduce_ndf
                if self.model.target.has_mask:
                    self.mask = torch.cat(
                        (
                            self.mask,
                            torch.zeros_like(con.reference_value, dtype=torch.bool),
                        )
                    )

    def L_up(self, Lup=None):
        if Lup is None:
            Lup = self.Lup
        self.L = min(1e9, self.L * Lup)

    def L_dn(self, Ldn=None):
        if Ldn is None:
            Ldn = self.Ldn
        self.L = max(1e-9, self.L / Ldn)

    def step(self, current_state=None) -> None:
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
            AP_config.ap_logger.info(f"h: {h.detach().cpu().numpy()}")

        self.update_Yp(h)
        loss = self.update_chi2()
        if self.verbose > 0:
            AP_config.ap_logger.info(f"LM loss: {loss.item()}")

        if self.iteration == 0:
            self.prev_Y[1] = self.current_Y
        self.loss_history.append(loss.detach().cpu().item())
        self.L_history.append(self.L)
        self.lambda_history.append(
            np.copy((self.current_state + h).detach().cpu().numpy())
        )

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
            lossmin = np.nanmin(self.loss_history[:-1])
            rho = self.rho(lossmin, loss, h)
            if self.verbose > 1:
                AP_config.ap_logger.debug(
                    f"LM loss: {loss.item()}, best loss: {np.nanmin(self.loss_history[:-1])}, loss diff: {np.nanmin(self.loss_history[:-1]) - loss.item()}, L: {self.L}"
                )
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
                if 0 < ((lossmin - loss) / loss) < self.relative_tolerance:
                    self._count_finish += 1
                else:
                    self._count_finish = 0
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

        if (
            (not self.use_broyden)
            or self.J is None
            or self.iteration < 2
            or "reset" in self.decision_history[-2:]
            or rho < self.epsilon4
            or self._count_reject > 0
            or self.iteration >= (2 * len(self.current_state))
            or self.decision_history[-1] == "nan"
        ):
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
                if (
                    self.save_steps is not None
                    and self.decision_history[-1] == "accept"
                ):
                    self.model.save(
                        os.path.join(
                            self.save_steps,
                            f"{self.model.name}_Iteration_{self.iteration:03d}.yaml",
                        )
                    )

                lam, L, loss = self.progress_history()

                # Check for convergence
                if (
                    self.decision_history.count("accept") > 2
                    and self.decision_history[-1] == "accept"
                    and L[-1] < 0.1
                    and ((loss[-2] - loss[-1]) / loss[-1])
                    < (self.relative_tolerance / 10)
                ):
                    self._count_converged += 1
                elif self.iteration >= self.max_iter:
                    self.message = (
                        self.message + f"fail max iterations reached: {self.iteration}"
                    )
                    break
                elif not torch.all(torch.isfinite(self.current_state)):
                    self.message = self.message + "fail non-finite step taken"
                    break
                elif (
                    self.L >= (1e9 - 1)
                    and self._count_reject >= 8
                    and not self.take_low_rho_step()
                ):
                    self.message = (
                        self.message
                        + "fail by immobility, unable to find improvement or even small bad step"
                    )
                    break
                if self._count_converged >= 3:
                    self.message = self.message + "success"
                    break
                lam, L, loss = self.accept_history()
                if len(loss) >= 10:
                    loss10 = np.array(loss[-10:])
                    if (
                        np.all(
                            np.abs((loss10[0] - loss10[-1]) / loss10[-1])
                            < self.relative_tolerance
                        )
                        and L[-1] < 0.1
                    ):
                        self.message = self.message + "success"
                        break
                    if (
                        np.all(
                            np.abs((loss10[0] - loss10[-1]) / loss10[-1])
                            < self.relative_tolerance
                        )
                        and L[-1] >= 0.1
                    ):
                        self.message = (
                            self.message
                            + "fail by immobility, possible bad area of parameter space."
                        )
                        break
        except KeyboardInterrupt:
            self.message = self.message + "fail interrupted"

        if self.message.startswith("fail") and self._count_finish > 0:
            self.message = (
                self.message
                + ". possibly converged to numerical precision and could not make a better step."
            )
        self.model.parameters.set_values(
            self.res(),
            as_representation=True,
            parameters_identity=self.fit_parameters_identity,
        )
        if self.verbose > 1:
            AP_config.ap_logger.info(
                f"LM Fitting complete in {time() - start_fit} sec with message: {self.message}"
            )

        return self

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

    @torch.no_grad()
    def undo_step(self) -> None:
        AP_config.ap_logger.info("undoing step, trying to recover")
        assert (
            self.decision_history.count("accept") >= 2
        ), "cannot undo with not enough accepted steps, retry with new parameters"
        assert len(self.decision_history) == len(self.lambda_history)
        assert len(self.decision_history) == len(self.L_history)
        found_accept = False
        for i in reversed(range(len(self.decision_history))):
            if not found_accept and self.decision_history[i] == "accept":
                found_accept = True
                continue
            if self.decision_history[i] != "accept":
                continue
            self.current_state = torch.tensor(
                self.lambda_history[i],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
            self.L = self.L_history[i] * self.Lup

    def take_low_rho_step(self) -> bool:
        for i in reversed(range(len(self.decision_history))):
            if "accept" in self.decision_history[i]:
                return False
            if self.rho_history[i] is not None and self.rho_history[i] > 0:
                if self.verbose > 0:
                    AP_config.ap_logger.info(
                        f"taking a low rho step for some progress: {self.rho_history[i]}"
                    )
                self.current_state = torch.tensor(
                    self.lambda_history[i],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                self.L = self.L_history[i]

                self.loss_history.append(self.loss_history[i])
                self.L_history.append(self.L)
                self.lambda_history.append(
                    np.copy((self.current_state).detach().cpu().numpy())
                )
                self.decision_history.append("low rho accept")
                self.rho_history.append(self.rho_history[i])

                with torch.no_grad():
                    self.update_Yp(torch.zeros_like(self.current_state))
                    self.prev_Y[0] = self.prev_Y[1]
                    self.prev_Y[1] = self.current_Y
                self.update_J_AD()
                self.update_hess()
                self.update_grad(self.prev_Y[1])
                self.iteration += 1
                self.count_reject = 0
                return True

    @torch.no_grad()
    def update_h(self) -> torch.Tensor:
        """Solves the LM update linear equation (H + L*I)h = G to determine
        the proposal for how to adjust the parameters to decrease the
        chi2.

        """
        h = torch.zeros_like(self.current_state)
        if self.iteration == 0:
            return h

        h = torch.linalg.solve(
            (
                self.hess
                + self.L**2
                * torch.eye(
                    len(self.grad), dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            )
            * (
                1
                + self.L**2
                * torch.eye(
                    len(self.grad), dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            ) ** 2
            / (1 + self.L**2),
            self.grad,
        )
        return h

    @torch.no_grad()
    def update_Yp(self, h):
        """
        Updates the current model values for each pixel
        """
        # Sample model at proposed state
        self.current_Y = self.model(
            parameters=self.current_state + h,
            as_representation=True,
            parameters_identity=self.fit_parameters_identity,
            window=self.fit_window,
        ).flatten("data")

        # Add constraint evaluations
        if self.constraints is not None:
            for con in self.constraints:
                self.current_Y = torch.cat((self.current_Y, con(self.model)))

    @torch.no_grad()
    def update_chi2(self):
        """
        Updates the chi squared / ndf value
        """
        # Apply mask if needed
        if self.model.target.has_mask:
            loss = (
                torch.sum(
                    ((self.Y - self.current_Y) ** 2 * self.W)[
                        torch.logical_not(self.mask)
                    ]
                )
                / self.ndf
            )
        else:
            loss = torch.sum((self.Y - self.current_Y) ** 2 * self.W) / self.ndf

        return loss

    def update_J_AD(self) -> None:
        """
        Update the jacobian using automatic differentiation, produces an accurate jacobian at the current state.
        """
        # Free up memory
        del self.J
        if "cpu" not in AP_config.ap_device:
            torch.cuda.empty_cache()

        # Compute jacobian on image
        self.J = self.model.jacobian(
            torch.clone(self.current_state).detach(),
            as_representation=True,
            parameters_identity=self.fit_parameters_identity,
            window=self.fit_window,
        ).flatten("data")

        # compute the constraint jacobian if needed
        if self.constraints is not None:
            for con in self.constraints:
                self.J = torch.cat((self.J, con.jacobian(self.model)))

        # Apply mask if needed
        if self.model.target.has_mask:
            self.J[self.mask] = 0.0

        # Note that the most recent jacobian was a full autograd jacobian
        self.full_jac = True

    def update_J_natural(self) -> None:
        """
        Update the jacobian using automatic differentiation, produces an accurate jacobian at the current state. Use this method to get the jacobian in the parameter space instead of representation space.
        """
        # Free up memory
        del self.J
        if "cpu" not in AP_config.ap_device:
            torch.cuda.empty_cache()

        # Compute jacobian on image
        self.J = self.model.jacobian(
            torch.clone(
                self.model.parameters.transform(
                    self.current_state,
                    to_representation=False,
                    parameters_identity=self.fit_parameters_identity,
                )
            ).detach(),
            as_representation=False,
            parameters_identity=self.fit_parameters_identity,
            window=self.fit_window,
        ).flatten("data")

        # compute the constraint jacobian if needed
        if self.constraints is not None:
            for con in self.constraints:
                self.J = torch.cat((self.J, con.jacobian(self.model)))

        # Apply mask if needed
        if self.model.target.has_mask:
            self.J[self.mask] = 0.0

        # Note that the most recent jacobian was a full autograd jacobian
        self.full_jac = False

    @torch.no_grad()
    def update_J_Broyden(self, h, Yp, Yph) -> None:
        """
        Use the Broyden update to approximate the new Jacobian tensor at the current state. Less accurate, but far faster.
        """

        # Update the Jacobian
        self.J = Broyden_step(self.J, h, Yp, Yph)

        # Apply mask if needed
        if self.model.target.has_mask:
            self.J[self.mask] = 0.0

        # compute the constraint jacobian if needed
        if self.constraints is not None:
            for con in self.constraints:
                self.J = torch.cat((self.J, con.jacobian(self.model)))

        # Note that the most recent jacobian update was with Broyden step
        self.full_jac = False

    @torch.no_grad()
    def update_hess(self) -> None:
        """
        Update the Hessian using the jacobian most recently computed on the image.
        """

        if isinstance(self.W, float):
            self.hess = torch.matmul(self.J.T, self.J)
        else:
            self.hess = torch.matmul(self.J.T, self.W.view(len(self.W), -1) * self.J)
        self.hess += self.epsilon5 * torch.eye(
            len(self.current_state),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    @property
    @torch.no_grad()
    def covariance_matrix(self) -> torch.Tensor:
        if self._covariance_matrix is not None:
            return self._covariance_matrix
        self.update_J_natural()
        self.update_hess()
        try:
            self._covariance_matrix = 2*torch.linalg.inv(self.hess)
        except:
            AP_config.ap_logger.warning(
                "WARNING: Hessian is singular, likely at least one model is non-physical. Will massage Hessian to continue but results should be inspected."
            )
            self.hess += torch.eye(
                len(self.grad), dtype=AP_config.ap_dtype, device=AP_config.ap_device
            ) * (torch.diag(self.hess) == 0)
            self._covariance_matrix = 2*torch.linalg.inv(self.hess)
        return self._covariance_matrix

    @torch.no_grad()
    def update_grad(self, Yph) -> None:
        """
        Update the gradient using the model evaluation on all pixels
        """
        self.grad = torch.matmul(self.J.T, self.W * (self.Y - Yph))

    @torch.no_grad()
    def rho(self, Xp, Xph, h) -> torch.Tensor:
        return (
            self.ndf
            * (Xp - Xph)
            / abs(
                torch.dot(
                    h,
                    self.L**2 * (torch.abs(torch.diag(self.hess) - self.epsilon5) * h)
                    + self.grad,
                )
            )
        )

    def accept_history(self) -> (List[np.ndarray], List[np.ndarray], List[float]):
        lambdas = []
        Ls = []
        losses = []

        for l in range(len(self.decision_history)):
            if "accept" in self.decision_history[l] and np.isfinite(
                self.loss_history[l]
            ):
                lambdas.append(self.lambda_history[l])
                Ls.append(self.L_history[l])
                losses.append(self.loss_history[l])
        return lambdas, Ls, losses

    def progress_history(self) -> (List[np.ndarray], List[np.ndarray], List[float]):
        lambdas = []
        Ls = []
        losses = []

        for l in range(len(self.decision_history)):
            if self.decision_history[l] == "accept":
                lambdas.append(self.lambda_history[l])
                Ls.append(self.L_history[l])
                losses.append(self.loss_history[l])
        return lambdas, Ls, losses

      
        

class LM_Constraint:
    """Add an arbitrary constraint to the LM optimization algorithm.

    Expresses a constraint between parameters in the LM optimization
    routine. Constraints may be used to bias parameters to have
    certain behaviour, for example you may require the radius of one
    model to be larger than that of another, or may require two models
    to have the same position on the sky. The constraints defined in
    this object are fuzzy constraints and so can be broken to some
    degree, the amount of constraint breaking is determined my how
    informative the data is and how strong the constraint weight is
    set. To create a constraint, first construct a function which
    takes as argument a 1D tensor of the model parameters and gives as
    output a real number (or 1D tensor of real numbers) which is zero
    when the constraint is satisfied and non-zero increasing based on
    how much the constraint is violated. For example:

    def example_constraint(P):
        return (P[1] - P[0]) * (P[1] > P[0]).int()

    which enforces that parameter 1 is less than parameter 0. Note
    that we do not use any control flow "if" statements and instead
    incorporate the condition through multiplication, this is
    important as it allows pytorch to compute derivatives through the
    expression and performs far faster on GPU since no communication
    is needed back and forth to handle the if-statement. Keep this in
    mind while constructing your constraint function. Also, make sure
    that any math operations are performed by pytorch so it can
    construct a computational graph. Bayond the requirement that the
    constraint be differentiable, there is no limitation on what
    constraints can be built with this system.

    Args:
      constraint_func (Callable[torch.Tensor, torch.Tensor]): python function which takes in a 1D tensor of parameters and generates real values in a tensor.
      constraint_args (Optional[tuple]): An optional tuple of arguments for the constraint function that will be unpacked when calling the function.
      weight (torch.Tensor): The weight of this constraint in the range (0,inf). Smaller values mean a stronger constraint, larger values mean a weaker constraint. Default 1.
      representation_parameters (bool): if the constraint_func expects the parameters in the form of their representation or their standard value. Default False
      out_len (int): the length of the output tensor by constraint_func. Default 1
      reference_value (torch.Tensor): The value at which the constraint is satisfied. Default 0.
      reduce_ndf (float): Amount by which to reduce the degrees of freedom. Default 0.

    """

    def __init__(
        self,
        constraint_func: Callable[[torch.Tensor, Any], torch.Tensor],
        constraint_args: tuple = (),
        representation_parameters: bool = False,
        out_len: int = 1,
        reduce_ndf: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        reference_value: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        self.constraint_func = constraint_func
        self.constraint_args = constraint_args
        self.representation_parameters = representation_parameters
        self.out_len = out_len
        self.reduce_ndf = reduce_ndf
        self.reference_value = torch.as_tensor(
            reference_value if reference_value is not None else torch.zeros(out_len),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        self.weight = torch.as_tensor(
            weight if weight is not None else torch.ones(out_len),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )

    def jacobian(self, model: "AutoPhot_Model"):
        jac = jacobian(
            lambda P: self.constraint_func(P, *self.constraint_args),
            model.parameters.get_vector(
                as_representation=self.representation_parameters
            ),
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

        return jac.reshape(-1, np.sum(model.parameters.vector_len()))

    def __call__(self, model: "AutoPhot_Model"):
        return self.constraint_func(
            model.parameters.get_vector(
                as_representation=self.representation_parameters
            ),
            *self.constraint_args,
        )
