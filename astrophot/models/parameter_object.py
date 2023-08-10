from typing import Optional
from copy import deepcopy

import torch

from ..utils.conversions.optimization import (
    boundaries,
    inv_boundaries,
    d_boundaries_dval,
    d_inv_boundaries_dval,
    cyclic_boundaries,
)
from .. import AP_config

__all__ = ["Parameter"]


class Parameter(object):
    """Object for storing model parameters that are to be optimized in the
    fitting procedure. A value and it's uncertainty are stored as well
    as meta information about the parameter. The meta information
    indicates if the parameter has any boundary values (upper or lower
    limits) and if the parameter has cyclic boundary conditions. The
    units for the parameter are generally also specified. The
    parameter tracks if it is "locked" meaning that it should not
    change during normal fitting conditions.

    For the purpose of optimization, the parameter object also stores
    a representation of it's value which is valid in the -inf to +inf
    range. This makes the parameter better behaved with most
    optimizers as they are free to explore the full real number
    line. Any time the value, or representation are updated, both
    numbers get updated automatically so they are always in sync.

    Properties:
        name: the name of the parameter [str]
        value: the value of the parameter for the sake of evaluating the model [float]
        representation: alternate view of the parameter which is defined on the -inf,inf range. Only specify value or representation, not both [float]
        uncertainty: 1 sigma uncertainty on the value of the parameter [float]
        limits: tuple of values specifying the valid range for this parameter, use None to mean no limit [tuple: (float, float), (float, None), (None, float), None]
        cyclic: boolean indicating if the parameter is cyclically defined. Note that cyclic parameters must specify limits [bool]
        locked: boolean indicating if the parameter should have a fixed value [bool]
        units: units for the value of the parameter. [str]
    """

    identity_list = []

    def __init__(self, name, value=None, **kwargs):
        self.name = name
        if "identity" in kwargs:
            self._identity = kwargs["identity"]
        else:
            self.identity = str(id(self))
        self._prof = None
        self._representation = None
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self.locked = kwargs.get("locked", False)
        self._representation = None
        self.set_value(value, override_locked=True)
        self.units = kwargs.get("units", "none")
        self._uncertainty = None
        self.uncertainty = kwargs.get("uncertainty", None)
        self.set_profile(kwargs.get("prof", None), override_locked=True)
        self.groups = kwargs.get("groups", set())
        self.to()

    @property
    def identity(self):
        return self._identity

    @identity.setter
    def identity(self, val):
        if val in Parameter.identity_list:
            c = 1
            while f"{val}c{c}" in Parameter.identity_list:
                c += 1
            val = f"{val}c{c}"
        self._identity = val
        Parameter.identity_list.append(val)

    def copy(self):
        return Parameter(
            name=self.name,
            value=self.value.clone(),
            limits=self.limits,
            cyclic=self.cyclic,
            locked=self.locked,
            units=self.units,
            uncertainty=self.uncertainty,
            prof=self.prof,
            groups=self.groups,
        )

    @property
    def representation(self):
        """The representation is the stored number (or tensor of numbers) for
        this parameter, it is what the optimizer sees and is defined
        in the range (-inf,+inf). This makes it well behaved during
        optimization. This is stored as a pytorch tensor which can
        track gradients.

        """
        return self.get_representation()

    @representation.setter
    def representation(self, rep):
        """Calls the set representation method, preserving locked behaviour."""
        self.set_representation(rep)

    @property
    def value(self):
        """The value of the parameter is what should be used in model
        evaluation or really for any typical purpose except in an
        optimizer.

        """
        return self.get_value()

    @value.setter
    def value(self, val):
        """
        Calls the value setting method, preserving locked behaviour
        """
        self.set_value(val)

    @property
    def locked(self):
        """If locked, the parameter cannot normally be updated and will no
        longer have "requires_grad" in pytorch.

        """
        return self._locked

    @locked.setter
    def locked(self, value):
        """
        updates the locked state of the parameter
        """
        self._locked = value

    @property
    def identities(self):
        return tuple(f"{self.identity}:{i}" for i in range(self.value.numel()))

    @property
    def names(self):
        return tuple(f"{self.name}:{i}" for i in range(self.value.numel()))

    @property
    def limits(self):
        try:
            return self._limits
        except AttributeError:
            return None

    @limits.setter
    def limits(self, val):
        if val is None:
            self._limits = None
        else:
            self._limits = (
                None
                if val[0] is None
                else torch.as_tensor(
                    val[0], dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
                None
                if val[1] is None
                else torch.as_tensor(
                    val[1], dtype=AP_config.ap_dtype, device=AP_config.ap_device
                ),
            )

    @property
    def uncertainty(self):
        """The uncertainty for the parameter is stored here, the uncertainty
        is for the value, not the representation.

        """
        return self.get_uncertainty()

    @uncertainty.setter
    def uncertainty(self, unc):
        """
        Calls the uncertainty setting method, preserving locked behaviour.
        """
        self.set_uncertainty(unc)

    @property
    def uncertainty_representation(self):
        """The uncertainty for the parameter is stored here, the uncertainty
        is for the value, not the representation.

        """
        if not self.cyclic and self.limits is not None:
            return self._uncertainty * d_boundaries_dval(self.value, self.limits)
        return self._uncertainty

    @uncertainty.setter
    def uncertainty_representation(self, unc):
        """
        Calls the uncertainty setting method, preserving locked behaviour.
        """
        self.set_uncertainty(unc, as_representation=True)

    @property
    def prof(self):
        return self._prof

    @prof.setter
    def prof(self, val):
        self.set_profile(val)

    @property
    def grad(self):
        """Returns the gradient for the representation of this parameter, if
        available.

        """
        return self._representation.grad

    def to(self, dtype=None, device=None):
        """
        updates the datatype or device of this parameter
        """
        if dtype is not None:
            dtype = AP_config.ap_dtype
        if device is not None:
            device = AP_config.ap_device
        if self._representation is not None:
            self._representation = self._representation.to(dtype=dtype, device=device)
        if self._uncertainty is not None:
            self._uncertainty = self._uncertainty.to(dtype=dtype, device=device)
        if self.prof is not None:
            self.prof = self.prof.to(dtype=dtype, device=device)
        return self

    def get_representation(self, index=None, identity=None):
        if self._representation is None:
            return None

        if identity is not None and self._representation.numel() > 1:
            index = int(identity[identity.find(":") + 1 :])

        if index is not None:
            return self._representation[index]

        return self._representation

    def get_value(self, index=None, identity=None):
        if self._representation is None:
            return None
        if self.cyclic:
            return cyclic_boundaries(
                self.get_representation(index=index, identity=identity), self.limits
            )
        if self.limits is None:
            return self.get_representation(index=index, identity=identity)
        return inv_boundaries(
            self.get_representation(index=index, identity=identity), self.limits
        )

    def rep_to_val(self, rep):
        if self.cyclic:
            return cyclic_boundaries(rep, self.limits)
        if self.limits is None:
            return rep
        return inv_boundaries(rep, self.limits)

    def val_to_rep(self, val):
        if self.cyclic:
            return cyclic_boundaries(val, self.limits)
        if self.limits is None:
            return val
        return boundaries(val, self.limits)

    def get_uncertainty(self, index=None, identity=None):
        if self._uncertainty is None:
            return None

        # Ensure the shape of uncertinty matches the value
        if (
            self._uncertainty.numel() == 1
            and self._representation is not None
            and self._representation.numel() > 1
        ):
            self._uncertainty = self._uncertainty * torch.ones_like(
                self._representation
            )

        if identity is not None and self._uncertainty.numel() > 1:
            index = int(identity[identity.find(":") + 1 :])

        if index is not None:
            return self._uncertainty[index]

        return self._uncertainty

    def set_uncertainty(
        self,
        uncertainty,
        override_locked=False,
        as_representation=False,
        index=None,
        identity=None,
    ):
        """Updates the the uncertainty of the value of the parameter. Only
        updates if the parameter is not locked.

        """
        if self.locked and not override_locked:
            return
        if uncertainty is None:
            self._uncertainty = None
            return

        # Choose correct index if setting by identity
        if identity is not None and self._representation.numel() > 1:
            index = int(identity[identity.find(":") + 1 :])

        # Ensure uncertainty is a tensor
        uncertainty = torch.as_tensor(
            uncertainty, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        # Set a single uncertainty from a tensor
        if index is not None:
            assert torch.all(
                uncertainty >= 0
            ), f"{self.name} Uncertainty should be a positive real value, not {uncertainty.item()}"
            if as_representation and not self.cyclic and self.limits is not None:
                self.uncertainty[index] = uncertainty * torch.abs(
                    d_inv_boundaries_dval(self.representation[index], self.limits)
                )
            else:
                self.uncertainty[index] = uncertainty
            return
        assert torch.all(
            uncertainty >= 0
        ), f"{self.name} Uncertainty should be a positive real value, not {uncertainty.detach().cpu().tolist()}"

        # Set the uncertainty variable
        if as_representation and not self.cyclic and self.limits is not None:
            self._uncertainty = uncertainty * d_inv_boundaries_dval(
                self.representation, self.limits
            )
        else:
            self._uncertainty = uncertainty

    def set_value(
        self,
        val,
        override_locked: bool = False,
        index: Optional[int] = None,
        identity: Optional[str] = None,
    ):
        """Set the value of the parameter. In fact this indirectly updates the
        representation for the parameter accoutning for any limits or
        cyclic boundaries are applied for this parameter.

        """
        if self.locked and not override_locked:
            return
        if identity is not None and self._representation.numel() > 1:
            index = int(identity[identity.find(":") + 1 :])
        if val is None:
            self._representation = None
        elif self.cyclic:
            self.set_representation(
                cyclic_boundaries(val, self.limits),
                override_locked=override_locked,
                index=index,
            )
        elif self.limits is None:
            self.set_representation(val, override_locked=override_locked, index=index)
        else:
            val = torch.as_tensor(
                val, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            if self.limits[0] is None:
                val = torch.clamp(val, max=self.limits[1] - 1e-3)
            elif self.limits[1] is None:
                val = torch.clamp(val, min=self.limits[0] + 1e-3)
            else:
                rng = self.limits[1] - self.limits[0]
                val = torch.clamp(
                    val,
                    min=self.limits[0]
                    + torch.min(1e-3 * torch.ones_like(rng), 1e-3 * rng),
                    max=self.limits[1]
                    - torch.min(1e-3 * torch.ones_like(rng), 1e-3 * rng),
                )
            self.set_representation(
                boundaries(val, self.limits),
                override_locked=override_locked,
                index=index,
            )

    def set_representation(
        self,
        rep,
        override_locked: bool = False,
        index: Optional[int] = None,
        identity: Optional[str] = None,
    ):
        """Update the representation for this parameter. Ensures that the
        representation is a pytorch tensor for optimization purposes.

        """
        if self.locked and not override_locked:
            return
        if identity is not None and self._representation.numel() > 1:
            index = int(identity[identity.find(":") + 1 :])
        if rep is None:
            self._representation = None
        elif index is None:
            self._representation = (
                rep.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
                if isinstance(rep, torch.Tensor)
                else torch.as_tensor(
                    rep, dtype=AP_config.ap_dtype, device=AP_config.ap_device
                )
            )
        else:
            self._representation[index] = torch.as_tensor(
                rep, dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )

    def get_state(self):
        """Return the values representing the current state of the parameter,
        this can be used to re-load the state later from memory.

        """
        state = {
            "name": self.name,
            "identity": self.identity,
        }
        if self.value is not None:
            state["value"] = self.value.detach().cpu().numpy().tolist()
        if self.units is not None:
            state["units"] = self.units
        if self.uncertainty is not None:
            state["uncertainty"] = self.uncertainty.detach().cpu().numpy().tolist()
        if self.locked:
            state["locked"] = self.locked
        if self.limits is not None:
            save_lim = []
            for i in [0, 1]:
                if self.limits[i] is None:
                    save_lim.append(None)
                elif self.limits[i].numel() == 1:
                    save_lim.append(self.limits[i].item())
                else:
                    save_lim.append(self.limits[i].detach().cpu().tolist())
            state["limits"] = tuple(save_lim)
        if self.cyclic:
            state["cyclic"] = self.cyclic
        if self.prof is not None:
            state["prof"] = self.prof.detach().cpu().tolist()

        return state

    def set_profile(self, prof, override_locked=False):
        if self.locked and not override_locked:
            return
        if prof is None:
            self._prof = None
            return
        self._prof = torch.as_tensor(
            prof, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )

    def set_state(self, state):
        """Update the state of the parameter given a state variable whcih
        holds all information about a variable.

        """
        self.name = state["name"]
        self._identity = state["identity"]
        self.units = state.get("units", None)
        self.limits = state.get("limits", None)
        self.cyclic = state.get("cyclic", False)
        self.set_uncertainty(state.get("uncertainty", None), override_locked = True)
        self.set_value(state.get("value", None), override_locked = True)
        self.set_profile(state.get("prof", None), override_locked = True)
        self.locked = state.get("locked", False)

    def __str__(self):
        """String representation of the parameter which indicates it's value
        along with uncertainty, units, limits, etc.

        """

        value = self.value.detach().cpu().tolist() if self.value is not None else "None"
        uncertainty = (
            self.uncertainty.detach().cpu().tolist()
            if self.uncertainty is not None
            else "None"
        )
        if self.limits is None:
            limits = None
        else:
            limits0 = (
                self.limits[0].detach().cpu().tolist()
                if self.limits[0] is not None
                else "None"
            )
            limits1 = (
                self.limits[1].detach().cpu().tolist()
                if self.limits[1] is not None
                else "None"
            )
            limits = (limits0, limits1)
        return f"{self.name}: {value} +- {uncertainty} [{self.units}{'' if self.locked is False else ', locked'}{'' if limits is None else (', ' + str(limits))}{'' if self.cyclic is False else ', cyclic'}]"

    def __iter__(self):
        """If the parameter has multiple values, it is posible to iterate over
        the values.

        """
        self.i = -1
        return self

    def __next__(self):
        self.i += 1
        if self.i < len(self.value):
            return self[self.i]
        else:
            raise StopIteration()

    def __getitem__(self, S):
        """If the parameter has multiple values, get the value at a given
        index.

        """
        if isinstance(S, int):
            return self.value[S]
        if isinstance(S, str):
            return self.value[int(S[S.rfind("|") + 1 :])]
        return self.value

    def __eq__(self, other):
        return self is other

    def __len__(self):
        """If the parameter has multiple values, this is the length of the
        parameter.

        """
        return len(self.value)
