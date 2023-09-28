from .base import Node

__all__ = ("Param_Unlock", "Param_SoftLimits")

class Param_Unlock(object):
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

class Param_SoftLimits(object):

    def __init__(self, param):
        self.param = param

    def __enter__(self, *args, **kwargs):
        self.original_setter = self.param._set_val_self
        self.param._set_val_self = self.param._soft_set_val_self
            
    def __exit__(self, *args, **kwargs):
        self.param._set_val_self = self.original_setter   


class Param_OverrideShape(object):
    """Temporarily allow writing values to parameters, with the wrong
    shape.

    Temporarily sets the shape of the parameter to None, this means
    that new values for a parameter can be written without concern for
    previous values/shapes of the parameter.

    """

    def __init__(self, param):
        self.param = param

    def __enter__(self):
        self.original_shape = self.param.shape
        self.param.shape = None
            
    def __exit__(self, *args, **kwargs):
        if self.param.shape is None:
            self.param.shape = self.original_shape
   
class Param_Mask(object):

    def __init__(self, param, mask):
        self.param = param
        self.mask = mask

    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass
