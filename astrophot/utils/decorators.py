from functools import wraps
import inspect
import warnings

import numpy as np

from ..image import (
    Image_List,
    Model_Image_List,
    Target_Image_List,
    Window_List,
)


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


def default_internal(func):
    """This decorator inspects the input parameters for a function which
    expects to receive `image` and `parameters` arguments. If either
    of these are not given, then the model can use its default values
    for the parameters assuming the `image` is the internal `target`
    object and the `parameters` are the internally stored parameters.

    """
    sig = inspect.signature(func)
    handles = sig.parameters.keys()

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        if "window" in handles:
            window = bound.arguments.get("window")
            if window is None:
                bound.arguments["window"] = self.window

        if "image" in handles:
            image = bound.arguments.get("image")
            if image is None:
                bound.arguments["image"] = self.target
            elif isinstance(image, Model_Image_List) and not isinstance(self.target, Image_List):
                for i, sub_image in enumerate(image):
                    if sub_image.target_identity == self.target.identity:
                        bound.arguments["image"] = sub_image
                        if "window" in bound.arguments and isinstance(
                            bound.arguments["window"], Window_List
                        ):
                            bound.arguments["window"] = bound.arguments["window"].window_list[i]
                        break
                else:
                    raise RuntimeError(f"{self.name} could not find matching image to sample with")

        if "target" in handles:
            target = bound.arguments.get("target")
            if target is None:
                bound.arguments["target"] = self.target
            elif isinstance(target, Target_Image_List) and not isinstance(self.target, Image_List):
                for sub_target in target:
                    if sub_target.identity == self.target.identity:
                        bound.arguments["target"] = sub_target
                        break
                else:
                    raise RuntimeError(
                        f"{self.name} could not find matching target to initialize with"
                    )

        return func(*bound.args, **bound.kwargs)

    return wrapper
