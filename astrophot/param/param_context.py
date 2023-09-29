from .base import Node

__all__ = ("Param_Unlock", "Param_SoftLimits", "Param_Mask")

class Param_Unlock:
    """Temporarily unlock a parameter.

    Context manager to unlock a parameter temporarily. Inside the
    context, the parameter will behave as unlocked regardless of its
    initial condition. Upon exiting the context, the parameter will
    return to it's previous locked state regardless of any changes
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
    """
    Temporarily allow writing parameter values outside limits.

    Values outside the limits will be quietly shift until they are within the boundaries of the parameter limits.
    """
    def __init__(self, param):
        self.param = param

    def __enter__(self, *args, **kwargs):
        self.original_setter = self.param._set_val_self
        self.param._set_val_self = self.param._soft_set_val_self
            
    def __exit__(self, *args, **kwargs):
        self.param._set_val_self = self.original_setter   


class Param_Mask:
    """
    Temporarily mask parameters.

    Select a subset of parameters to be used through the "vector" interface of the DAG.
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
