import torch
import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries

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

    """
    
    def __init__(self, name, value = None, **kwargs):

        self.name = name
        
        self.dtype = kwargs.get("dtype", torch.float64)
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self._locked = False
        self.set_value(value, override_locked = True)
        self.locked = kwargs.get("locked", False)
        self.units = kwargs.get("units", "none")
        self._uncertainty = kwargs.get("uncertainty", None)
        
    @property
    def representation(self):
        """The representation is the stored number (or array of numbers) for
        this parameter, it is what the optimizer sees and is defined
        in the range (-inf,+inf). This makes it well behaved during
        optimization. This is stored as a pytorch tensor which can
        track gradients.

        """
        return self._representation
    @representation.setter
    def representation(self, rep):
        self.set_representation(rep)
    @property
    def value(self):
        """The value of the parameter is what should be used in model
        evaluation or really for any typical purpose except in an
        optimizer.

        """
        if self._representation is None:
            return None
        if self.cyclic:
            return cyclic_boundaries(self._representation, self.limits)
        if self.limits is None:
            return self._representation
        return inv_boundaries(self._representation, self.limits)
    @value.setter
    def value(self, val):
        self.set_value(val)
    @property
    def locked(self):
        """If locked, the parameter cannot normally be updated and will no
        longer have "require_grad" in pytorch.

        """
        return self._locked
    @locked.setter
    def locked(self, value):
        self._locked = value
        # if self._representation is not None:
        #     self._representation.requires_grad = not self._locked
    @property
    def uncertainty(self):
        """The uncertainty for the parameter is stored here, the uncertainty
        is for the value, not the representation. # fixme is this best?

        """
        return self._uncertainty
    @uncertainty.setter
    def uncertainty(self, unc):
        self.set_uncertainty(unc)
    @property
    def grad(self):
        """Returns the gradient for the representation of this parameter, if
        available.

        """
        return self._representation.grad

    def to(self, dtype = None, device = None):
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        if self._representation is not None:
            self._representation = self._representation.to(dtype = self.dtype, device = self.device)
        return self
    
    def set_uncertainty(self, uncertainty, override_locked = False):
        """Updates the value for the uncertainty of the value of the
        parameter. Only updates if the parameter is not locked.

        """
        if self.locked and not override_locked:
            return
        if np.any(uncertainty < 0):
            raise ValueError(f"{name} Uncertainty should be a positive real value, not {uncertainty}")
        self._uncertainty = uncertainty

    def set_value(self, val, override_locked = False, index = None):
        """Set the value of the parameter. In fact this indirectly updates
        the representation for the parameter if any limits or cyclic
        boundaries are applied for this parameter.

        """
        if self.locked and not override_locked:
            return
        if val is None:
            self._representation = None
        elif self.cyclic:
            self.set_representation(cyclic_boundaries(val, self.limits), override_locked = override_locked, index = index)
        elif self.limits is None:
            self.set_representation(val, override_locked = override_locked, index = index)
        else:
            self.set_representation(boundaries(val, self.limits), override_locked = override_locked, index = index)
        
    def set_representation(self, rep, override_locked = False, index = None):
        """Update the representation for this parameter. Ensures that the
        representation is a pytorch tensor for optimization purposes.

        """
        if self.locked and not override_locked:
            return
        if rep is None:
            self._representation = None
        elif index is None:
            self._representation = rep.to(dtype = self.dtype, device = self.device) if isinstance(rep, torch.Tensor) else torch.tensor(rep, dtype = self.dtype, device = self.device)
        else:
            self._representation[index] = rep
        # if self._representation is not None:
        #     self._representation.requires_grad = not self.locked

    def get_state(self):
        
        state = {
            "name": self.name,
        }
        if self.value is not None:
            state["value"] = self.value.detach().cpu().numpy().tolist()
        if self.units is not None:
            state["units"] = self.units
        if self.uncertainty is not None:
            state["uncertainty"] = np.array(self.uncertainty).tolist()
        if self.locked:
            state["locked"] = self.locked
        if self.limits is not None:
            state["limits"] = self.limits
        if self.cyclic:
            state["cyclic"] = self.cyclic
            
        return state
    
    def update_state(self, state):
        self.name = state["name"]
        self.units = state.get("units", None)
        self.limits = state.get("limits", None)
        self.cyclic = state.get("cyclic", False)
        self.uncertainty = state.get("uncertainty", None)
        self.value = state.get("value", None)
        self.locked = state.get("locked", False)
            
    def __str__(self):
        """
        String representation of the parameter which indicates it's value along with uncertainty, units, limits, etc.
        """
        return f"{self.name}: {self.value} +- {self.uncertainty} [{self.units}{'' if self.locked is False else ', locked'}{'' if self.limits is None else (', ' + str(self.limits))}{'' if self.cyclic is False else ', cyclic'}]"

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
            return self.value[int(S[S.rfind("|")+1:])]
        return self.value
    
    def __len__(self):
        """If the parameter has multiple values, this is the length of the
        parameter.

        """
        return len(self.value)    
