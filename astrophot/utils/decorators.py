from functools import wraps
import warnings

import numpy as np


class classproperty:
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


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
        warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
        result = func(*args, **kwargs)
        np.seterr(**old_settings)
        warnings.filterwarnings("default", category=np.exceptions.VisibleDeprecationWarning)
        return result

    return wrapped


def combine_docstrings(cls):
    combined_docs = [cls.__doc__ or ""]
    for base in cls.__bases__:
        if base.__doc__:
            combined_docs.append(f"\n[UNIT {base.__name__}]\n\n{base.__doc__}")
    cls.__doc__ = "\n".join(combined_docs).strip()
    return cls
