from functools import wraps
import inspect
import warnings

import numpy as np


def ignore_numpy_warnings(func):
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
