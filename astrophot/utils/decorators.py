from functools import wraps
import warnings
from inspect import cleandoc

import numpy as np

__all__ = ("classproperty", "ignore_numpy_warnings", "combine_docstrings")


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
    try:
        combined_docs = [cleandoc(cls.__doc__)]
    except AttributeError:
        combined_docs = []
    for base in cls.__bases__:
        if base.__doc__:
            combined_docs.append(f"\n\n> SUBUNIT {base.__name__}\n\n{cleandoc(base.__doc__)}")
    cls.__doc__ = "\n".join(combined_docs).strip()
    return cls
