from . import config, models, plots, utils, fit, image, errors
from .param import forward, Param, Module

from .image import (
    Image,
    ImageList,
    TargetImage,
    TargetImageList,
    SIPModelImage,
    SIPTargetImage,
    CMOSModelImage,
    CMOSTargetImage,
    JacobianImage,
    JacobianImageList,
    PSFImage,
    ModelImage,
    ModelImageList,
    Window,
    WindowList,
)
from .models import Model
from .backend_obj import backend, ArrayLike

try:
    from ._version import version as VERSION  # noqa
except ModuleNotFoundError:
    VERSION = "0.0.0"
    print(
        "WARNING: AstroPhot version number not found. This is likely because you are running AstroPhot from a source directory."
    )


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
    "TargetImage",
    "TargetImageList",
    "SIPModelImage",
    "SIPTargetImage",
    "CMOSModelImage",
    "CMOSTargetImage",
    "JacobianImage",
    "JacobianImageList",
    "PSFImage",
    "ModelImage",
    "ModelImageList",
    "Window",
    "WindowList",
    "plots",
    "utils",
    "fit",
    "forward",
    "Param",
    "errors",
    "Module",
    "config",
    "backend",
    "ArrayLike",
    "__version__",
    "__author__",
    "__email__",
)
