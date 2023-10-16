from functools import wraps
import inspect
import warnings

import numpy as np


def ignore_numpy_warnings(func):
    """This decorator is used to turn off numpy warnings. This should
    only be used in initialize scripts which often run heuristic code
    to determine initial parameter values. These heuristics may
    encounter log(0) or sqrt(-1) or other numerical artifacts and
    should handle them before returning. This decorator simply cleans
    up that processes to minimize clutter in the output.

    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        old_settings = np.seterr(all="ignore")
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        result = func(*args, **kwargs)
        np.seterr(**old_settings)
        warnings.filterwarnings("default", category=np.VisibleDeprecationWarning)
        return result

    return wrapped


def default_internal(func):
    """This decorator inspects the input parameters for a function which
    expects to recieve `image` and `parameters` arguments. If either
    of these are not given, then the model can use its default values
    for the parameters assuming the `image` is the internal `target`
    object and the `parameters` are the internally stored parameters.

    """
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        if bound.arguments.get("image") is None:
            bound.arguments["image"] = self.target
        if bound.arguments.get("parameters") is None:
            bound.arguments["parameters"] = self.parameters

        return func(*bound.args, **bound.kwargs)

    return wrapper
