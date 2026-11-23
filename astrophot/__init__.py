from . import config, models, utils, fit, image, errors
from .config import set_logger
from .param import forward, Param, Module

from .image import (
    Image,
    ImageList,
    ImageBatchMixin,
    TargetImage,
    TargetImageList,
    TargetImageBatch,
    SIPModelImage,
    SIPTargetImage,
    CMOSModelImage,
    CMOSTargetImage,
    JacobianImage,
    JacobianImageList,
    JacobianImageBatch,
    PSFImage,
    ModelImage,
    ModelImageList,
    ModelImageBatch,
    Window,
    WindowList,
    WindowBatch,
)
from .models import Model
from .backend_obj import backend, ArrayLike

try:
    from ._version import version as VERSION  # noqa
except ModuleNotFoundError:
    VERSION = "0.0.0"
    config.logger.warning(
        "AstroPhot version number not found. This is likely because you are running AstroPhot from a source directory."
    )


def __getattr__(name):
    # matplotlib is slow to import, so ``astrophot.plots`` is loaded on first use
    if name == "plots":
        from importlib import import_module

        return import_module(".plots", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# meta data
__version__ = VERSION
__author__ = "Connor Stone"
__email__ = "connorstone628@gmail.com"

__all__ = (
    "models",
    "image",
    "Model",
    "Image",
    "ImageList",
    "ImageBatchMixin",
    "TargetImage",
    "TargetImageList",
    "TargetImageBatch",
    "SIPModelImage",
    "SIPTargetImage",
    "CMOSModelImage",
    "CMOSTargetImage",
    "JacobianImage",
    "JacobianImageList",
    "JacobianImageBatch",
    "PSFImage",
    "ModelImage",
    "ModelImageList",
    "ModelImageBatch",
    "Window",
    "WindowList",
    "WindowBatch",
    "plots",
    "utils",
    "fit",
    "forward",
    "Param",
    "errors",
    "Module",
    "config",
    "set_logger",
    "backend",
    "ArrayLike",
    "__version__",
    "__author__",
    "__email__",
)
