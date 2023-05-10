from functools import wraps
import numpy as np
import warnings

def ignore_numpy_warnings(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        old_settings = np.seterr(all='ignore')
        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        result = func(*args, **kwargs)
        np.seterr(**old_settings)
        warnings.filterwarnings('default', category=np.VisibleDeprecationWarning)
        return result
    return wrapped
