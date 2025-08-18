import argparse
import requests
import torch
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


def run_from_terminal() -> None:
    """
    Running from terminal no longer supported. This is only used for convenience to download the tutorials.

    """
    config.logger.debug("running from the terminal, not sure if it will catch me.")
    parser = argparse.ArgumentParser(
        prog="astrophot",
        description="Fast and flexible astronomical image photometry package. For the documentation go to: https://astrophot.readthedocs.io",
        epilog="Please see the documentation or contact connor stone (connorstone628@gmail.com) for further assistance.",
    )
    parser.add_argument(
        "filename",
        nargs="?",
        metavar="configfile",
        help="the path to the configuration file. Or just 'tutorial' to download tutorials.",
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="astrophot",
    #     choices=["astrophot", "galfit"],
    #     metavar="format",
    #     help="The type of configuration file being being provided. One of: astrophot, galfit.",
    # )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="print the current AstroPhot version to screen",
    )
    # parser.add_argument(
    #     "--log",
    #     type=str,
    #     metavar="logfile.log",
    #     help="set the log file name for AstroPhot. use 'none' to suppress the log file.",
    # )
    # parser.add_argument(
    #     "-q",
    #     action="store_true",
    #     help="quiet flag to stop command line output, only print to log file",
    # )
    # parser.add_argument(
    #     "--dtype",
    #     type=str,
    #     choices=["float64", "float32"],
    #     metavar="datatype",
    #     help="set the float point precision. Must be one of: float64, float32",
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     choices=["cpu", "gpu"],
    #     metavar="device",
    #     help="set the device for AstroPhot to use for computations. Must be one of: cpu, gpu",
    # )

    args = parser.parse_args()

    if args.log is not None:
        config.set_logging_output(
            stdout=not args.q, filename=None if args.log == "none" else args.log
        )
    elif args.q:
        config.set_logging_output(stdout=not args.q, filename="AstroPhot.log")

    if args.dtype is not None:
        config.DTYPE = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device is not None:
        config.DEVICE = "cpu" if args.device == "cpu" else "cuda:0"

    if args.filename is None:
        raise RuntimeError(
            "Please pass a config file to astrophot. See 'astrophot --help' for more information, or go to https://astrophot.readthedocs.io"
        )
    if args.filename in ["tutorial", "tutorials"]:
        tutorials = [
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/GettingStarted.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/GroupModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/ModelZoo.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/JointModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/FittingMethods.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/CustomModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/BasicPSFModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/AdvancedPSFModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/source/tutorials/ConstrainedModels.ipynb",
        ]
        for url in tutorials:
            try:
                R = requests.get(url)
                with open(url[url.rfind("/") + 1 :], "w") as f:
                    f.write(R.text)
            except:
                print(
                    f"WARNING: couldn't find tutorial: {url[url.rfind('/')+1:]} check internet connection"
                )

        config.logger.info("collected the tutorials")
    else:
        raise ValueError(f"Unrecognized request")


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
    "run_from_terminal",
    "__version__",
    "__author__",
    "__email__",
)
