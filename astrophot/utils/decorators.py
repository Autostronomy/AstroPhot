from functools import wraps
import re
import warnings
from inspect import cleandoc
import caskade as ck

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
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=ck.InvalidValueWarning)
        result = func(*args, **kwargs)
        np.seterr(**old_settings)
        warnings.filterwarnings("default", category=np.exceptions.VisibleDeprecationWarning)
        warnings.filterwarnings("default", category=DeprecationWarning)
        warnings.filterwarnings("default", category=ck.InvalidValueWarning)
        return result

    return wrapped


def _parse_docstring(doc):
    """Parse a docstring into (body, params_dict).

    Handles both RST ``:param name: desc`` format and markdown
    ``**Parameters:**`` / ``**Options:**`` format.
    Returns body text (params/options removed) and an ordered dict of {name: desc}.
    """
    params = {}

    # --- RST-style :param name: desc (possibly multi-line) ---
    # Split on ":param " boundaries so multi-line descriptions are captured correctly.
    for m in re.finditer(
        r':param\s+(\w+):\s*((?:(?!\n:(?:param|type)\s).)*)',
        doc,
        re.DOTALL,
    ):
        params[m.group(1)] = m.group(2).strip()

    # --- Markdown-style **Parameters:** and **Options:** sections ---
    for section_match in re.finditer(
        r'\*\*(?:Parameters|Options):\*\*\n((?:[ \t]*-[^\n]*\n?)+)',
        doc,
        re.MULTILINE,
    ):
        for item in re.finditer(
            r'^[ \t]*-\s+`(\w+)`\s*:\s*(.+?)(?=\n[ \t]*-|\Z)',
            section_match.group(1),
            re.DOTALL | re.MULTILINE,
        ):
            params[item.group(1)] = item.group(2).strip()

    # Strip :param/:type entries from body (handle multi-line descriptions)
    body = re.sub(r':(?:param|type)\s+\w+:\s*(?:(?!\n:(?:param|type)\s).)*', '', doc, flags=re.DOTALL)
    # Strip markdown Parameters/Options sections from body
    body = re.sub(r'\*\*(?:Parameters|Options):\*\*\n(?:[ \t]*-[^\n]*\n?)+', '', body)
    body = body.strip()

    return body, params


def combine_docstrings(cls):
    """Combine docstrings from a class and all of its base classes.

    Finds all ``:param`` entries and markdown ``**Parameters:**`` /
    ``**Options:**`` sections from every class in the MRO and merges
    them into a single consolidated ``:param`` list so that Sphinx
    autodoc renders them correctly.
    """
    # Collect params from full MRO (base → derived so derived class wins)
    all_params = {}
    for klass in reversed(cls.__mro__):
        if klass is object:
            continue
        if not klass.__doc__:
            continue
        _, klass_params = _parse_docstring(cleandoc(klass.__doc__))
        all_params.update(klass_params)

    # Build combined body: cls body + direct-base bodies
    try:
        main_body, _ = _parse_docstring(cleandoc(cls.__doc__))
    except (AttributeError, TypeError):
        main_body = ""

    for base in cls.__bases__:
        if base is object or not base.__doc__:
            continue
        base_body, _ = _parse_docstring(cleandoc(base.__doc__))
        if base_body:
            main_body += f"\n\n.. rubric:: {base.__name__}\n\n{base_body}"

    # Append merged parameter list
    if all_params:
        model_params = set()
        try:
            model_params = set(cls.parameter_specs.keys())
        except Exception:
            model_params = set()
        for name, desc in list(all_params.items()):
            if name in model_params and "[model param]" not in desc:
                all_params[name] = f"{desc} [model param]"
        main_body += "\n\n" + "\n".join(
            f":param {name}: {desc}" for name, desc in all_params.items()
        )

    cls.__doc__ = main_body.strip()
    return cls
