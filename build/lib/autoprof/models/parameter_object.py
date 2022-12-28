import torch
import numpy as np
from ..utils.conversions.optimization import boundaries, inv_boundaries, d_boundaries_dval, d_inv_boundaries_dval, cyclic_boundaries

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
        dtype: the data type for the value and representation [torch.dtype object]
        device: the computational device with which to associate the parameter, either CPU or the GPU [str: "cpu" or name of GPU typically "cuda:0"]
    """
    
    def __init__(self, name, value = None, **kwargs):
        self.name = name
        
        self.dtype = kwargs.get("dtype", torch.float64)
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self._representation = None
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self.locked = kwargs.get("locked", False)
        self._representation = None
        self.set_value(value, override_locked = True)
        self.requires_grad = kwargs.get("requires_grad", False)
        self.units = kwargs.get("units", "none")
        self._uncertainty = None
        self.uncertainty = kwargs.get("uncertainty", None)
        self.prof = None
        self.set_profile(kwargs.get("prof", None))
        self.to(dtype = self.dtype, device = self.device)
       
    @property
    def representation(self):
        """The representation is the stored number (or tensor of numbers) for
        this parameter, it is what the optimizer sees and is defined
        in the range (-inf,+inf). This makes it well behaved during
        optimization. This is stored as a pytorch tensor which can
        track gradients.

        """
        return self._representation
    @representation.setter
    def representation(self, rep):
        """Calls the set representation method, preserving locked behaviour.

        """
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
    def uncertainty(self):
        """The uncertainty for the parameter is stored here, the uncertainty
        is for the value, not the representation. 

        """
        return self._uncertainty
    @uncertainty.setter
    def uncertainty(self, unc):
        """
        Calls the uncertainty setting method, preserving locked behaviour.
        """
        self.set_uncertainty(unc)
    @property
    def requires_grad(self):
        if self._representation is None:
            return False
        return self._representation.requires_grad
    @requires_grad.setter
    def requires_grad(self, val):
        assert isinstance(val, bool)
        if self._representation is not None and not (self._representation.requires_grad is val):
            self._representation = self._representation.detach()
            self._representation.requires_grad = val
            
    @property
    def grad(self):
        """Returns the gradient for the representation of this parameter, if
        available.

        """
        return self._representation.grad

    def to(self, dtype = None, device = None):
        """
        updates the datatype or device of this parameter
        """
        if dtype is not None:
            self.dtype = dtype
        if device is not None:
            self.device = device
        if self._representation is not None:
            self._representation = self._representation.to(dtype = self.dtype, device = self.device)
        if self._uncertainty is not None:
            self._uncertainty = self._uncertainty.to(dtype = self.dtype, device = self.device)
        if self.prof is not None:
            self.prof = self.prof.to(dtype = self.dtype, device = self.device)
        return self
    
    def set_uncertainty(self, uncertainty, override_locked = False, as_representation = False):
        """Updates the the uncertainty of the value of the parameter. Only
        updates if the parameter is not locked.

        """
        if self.locked and not override_locked:
            return
        if uncertainty is None:
            self._uncertainty = None
            return
        uncertainty = torch.as_tensor(uncertainty, dtype = self.dtype, device = self.device)
        if torch.any(uncertainty < 0):
            raise ValueError(f"{self.name} Uncertainty should be a positive real value, not {uncertainty}")
        if as_representation and not self.cyclic and self.limits is not None:
            self._uncertainty = uncertainty * d_inv_boundaries_dval(self.representation, self.limits)
        else:
            self._uncertainty = uncertainty

    def set_value(self, val, override_locked = False, index = None):
        """Set the value of the parameter. In fact this indirectly updates the
        representation for the parameter accoutning for any limits or
        cyclic boundaries are applied for this parameter.

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
            val = torch.as_tensor(val, dtype = self.dtype, device = self.device)
            if self.limits[0] is None:
                val = torch.clamp(val, max = self.limits[1] - 1e-3)
            elif self.limits[1] is None:
                val = torch.clamp(val, min = self.limits[0] + 1e-3)
            else:
                rng = self.limits[1] - self.limits[0]
                val = torch.clamp(val, min = self.limits[0] + min(1e-3, 1e-3 * rng), max = self.limits[1] - min(1e-3, 1e-3 * rng))
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
            self._representation = rep.to(dtype = self.dtype, device = self.device) if isinstance(rep, torch.Tensor) else torch.as_tensor(rep, dtype = self.dtype, device = self.device)
        else:
            self._representation[index] = rep
        
    def get_state(self):
        """Return the values representing the current state of the parameter,
        this can be used to re-load the state later from memory.

        """
        state = {
            "name": self.name,
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
            state["limits"] = self.limits
        if self.cyclic:
            state["cyclic"] = self.cyclic
        if self.prof is not None:
            state["prof"] = self.prof.detach().cpu().numpy().tolist()
            
        return state

    def set_profile(self, prof, override_locked = False):
        if self.locked and not override_locked:
            return
        if prof is None:
            self.prof = None
            return
        self.prof = torch.as_tensor(prof, dtype = self.dtype, device = self.device)
            
    def update_state(self, state):
        """Update the state of the parameter given a state variable whcih
        holds all information about a variable.

        """
        self.name = state["name"]
        self.units = state.get("units", None)
        self.limits = state.get("limits", None)
        self.cyclic = state.get("cyclic", False)
        self.uncertainty = state.get("uncertainty", None)
        self.value = state.get("value", None)
        self.locked = state.get("locked", False)
        self.prof = state.get("prof", None)
        
    def __str__(self):
        """String representation of the parameter which indicates it's value
        along with uncertainty, units, limits, etc.

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

    def __eq__(self, other):
        return self is other
    
    def __len__(self):
        """If the parameter has multiple values, this is the length of the
        parameter.

        """
        return len(self.value)
