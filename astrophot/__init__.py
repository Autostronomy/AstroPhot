import sys
import argparse
import requests
import torch
from .parse_config import galfit_config, basic_config
from . import models, image, plots, utils, fit, param, AP_config

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
    Execute AstroPhot from the command line with various options.

    This function uses the `argparse` module to parse command line arguments and execute the appropriate functionality.
    It accepts the following arguments:

    - `filename`: the path to the configuration file. Or just 'tutorial' to download tutorials.
    - `--config`: the type of configuration file being provided. One of: astrophot, galfit.
    - `-v`, `--version`: print the current AstroPhot version to screen.
    - `--log`: set the log file name for AstroPhot. Use 'none' to suppress the log file.
    - `-q`: quiet flag to stop command line output, only print to log file.
    - `--dtype`: set the float point precision. Must be one of: float64, float32.
    - `--device`: set the device for AstroPhot to use for computations. Must be one of: cpu, gpu.

    If the `filename` argument is not provided, it raises a `RuntimeError`.
    If the `filename` argument is `tutorial` or `tutorials`,
    it downloads tutorials from various URLs and saves them locally.

    This function logs messages using the `AP_config` module,
    which sets the logging output based on the `--log` and `-q` arguments.
    The `dtype` and `device` of AstroPhot can also be set using the `--dtype` and `--device` arguments, respectively.

    Returns:
        None

    """
    AP_config.ap_logger.debug("running from the terminal, not sure if it will catch me.")
    parser = argparse.ArgumentParser(
        prog="astrophot",
        description="Fast and flexible astronomical image photometry package. For the documentation go to: https://github.com/Autostronomy/AstroPhot",
        epilog="Please see the documentation or contact connor stone (connorstone628@gmail.com) for further assistance.",
    )
    parser.add_argument(
        "filename",
        nargs="?",
        metavar="configfile",
        help="the path to the configuration file. Or just 'tutorial' to download tutorials.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="astrophot",
        choices=["astrophot", "galfit"],
        metavar="format",
        help="The type of configuration file being being provided. One of: astrophot, galfit.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="print the current AstroPhot version to screen",
    )
    parser.add_argument(
        "--log",
        type=str,
        metavar="logfile.log",
        help="set the log file name for AstroPhot. use 'none' to suppress the log file.",
    )
    parser.add_argument(
        "-q",
        action="store_true",
        help="quiet flag to stop command line output, only print to log file",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float64", "float32"],
        metavar="datatype",
        help="set the float point precision. Must be one of: float64, float32",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        metavar="device",
        help="set the device for AstroPhot to use for computations. Must be one of: cpu, gpu",
    )

    args = parser.parse_args()

    if args.log is not None:
        AP_config.set_logging_output(
            stdout=not args.q, filename=None if args.log == "none" else args.log
        )
    elif args.q:
        AP_config.set_logging_output(stdout=not args.q, filename="AstroPhot.log")

    if args.dtype is not None:
        AP_config.dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device is not None:
        AP_config.device = "cpu" if args.device == "cpu" else "cuda:0"

    if args.filename is None:
        raise RuntimeError(
            "Please pass a config file to astrophot. See 'astrophot --help' for more information, or go to https://Autostronomy.github.io/AstroPhot/getting_started.html"
        )
    if args.filename in ["tutorial", "tutorials"]:
        tutorials = [
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/GettingStarted.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/GroupModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/ModelZoo.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/JointModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/FittingMethods.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/CustomModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/AdvancedPSFModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/ConstrainedModels.ipynb",
            "https://raw.github.com/Autostronomy/AstroPhot/main/docs/tutorials/simple_config.py",
        ]
        for url in tutorials:
            try:
                R = requests.get(url)
                with open(url[url.rfind("/") + 1 :], "w") as f:
                    f.write(R.text)
            except:
                print(
                    f"WARNING: couldn't find tutorial: {url[url.rfind('/')+1:]} check internet conection"
                )

        AP_config.ap_logger.info("collected the tutorials")
    elif args.config == "astrophot":
        basic_config(args.filename)
    elif args.config == "galfit":
        galfit_config(args.filename)
    else:
        raise ValueError(
            f"Unrecognized configuration file format {args.config}. Should be one of: astrophot, galfit"
        )
