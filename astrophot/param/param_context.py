from .base import Node

__all__ = ("Param_Unlock", "Param_SoftLimits", "Param_Mask")

class Param_Unlock:
    """Temporarily unlock a parameter.

    Context manager to unlock a parameter temporarily. Inside the
    context, the parameter will behave as unlocked regardless of its
    initial condition. Upon exiting the context, the parameter will
    return to its previous locked state regardless of any changes
    made by the user to the lock state.

    """

    def __init__(self, param = None):
        self.param = param

    def __enter__(self):
        if self.param is None:
            Node.global_unlock = True
        else:
            self.original_locked = self.param.locked
            self.param.locked = False
            
    def __exit__(self, *args, **kwargs):
        if self.param is None:
            Node.global_unlock = False
        else:
            self.param.locked = self.original_locked

class Param_SoftLimits:
    """Temporarily allow writing parameter values outside limits.

    Values outside the limits will be quietly (no error/warning
    raised) shifted until they are within the boundaries of the
    parameter limits. Since the limits are non-inclusive, the soft
    limits will actually move a parameter by 0.001 into the parameter
    range. For example the axis ratio ``q`` has limits from (0,1) so
    if one were to write: ``q.value = 2`` then the actual value that
    gets written would be ``0.999``.

    Cyclic parameters are not affected by this, any value outside the
    range is always (Param_SoftLimits context or not) wrapped back
    into the range using modulo arithmetic.

    """
    def __init__(self, param):
        self.param = param

    def __enter__(self, *args, **kwargs):
        self.original_setter = self.param._set_val_self
        self.param._set_val_self = self.param._soft_set_val_self
            
    def __exit__(self, *args, **kwargs):
        self.param._set_val_self = self.original_setter   


class Param_Mask:
    """Temporarily mask parameters.

    Select a subset of parameters to be used through the "vector"
    interface of the DAG. The context is initialized with a
    Parameter_Node object (``P``) and a torch tensor (``M``) where the
    size of the mask should be equal to the current vector
    representation of the parameter (``M.numel() ==
    P.vector_values().numel()``). The mask tensor should be of
    ``torch.bool`` dtype where ``True`` indicates to keep using that
    parameter and ``False`` indicates to hide that parameter value.

    Note that ``Param_Mask`` contexts can be nested and will behave
    accordingly (the mask tensor will need to match the vector size
    within the previous context). As an example, imagine there is a
    parameter node ``P`` which has five sub-nodes each with a single
    value, one could nest contexts like::

      M1 = torch.tensor((1,1,0,1,0), dtype = torch.bool)
      with Param_Mask(P, M1):
        # Now P behaves as if it only has 3 elements
        M2 = torch.tensor([0,1,1], dtype = torch.bool)
        with Param_Mask(P, M2):
          # Now P behaves as if it only has 2 elements
          P.vector_values() # returns tensor with 2 elements

    """
    def __init__(self, param, new_mask):
        self.param = param
        self.new_mask = new_mask

    def __enter__(self):

        self.old_mask = self.param.vector_mask()
        self.mask = self.param.vector_mask()
        self.mask[self.mask.clone()] = self.new_mask
        self.param.vector_set_mask(self.mask)

    def __exit__(self, *args, **kwargs):
        self.param.vector_set_mask(self.old_mask)
