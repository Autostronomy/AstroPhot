import sys
import argparse
from . import models, image, plots, utils, fit, AP_config

# meta data
__version__ = "0.4.0"
__author__ = "Connor Stone"
__email__ = "connorstone628@gmail.com"

def run_from_terminal():
    AP_config.ap_logger.debug("running from the terminal, not sure if it will catch me.")
    parser = argparse.ArgumentParser(
        prog = "autoprof",
        description = "Fast and flexible astronomical image photometry package. For the documentation go to: https://github.com/ConnorStoneAstro/AutoProf",
        epilog = "Please see the documentation or contact connor stone (connorstone628@gmail.com) for further assistance."
    )
    parser.add_argument(
        "filename",
        nargs = "?",
        metavar = "configfile",
        help = "the path to the configuration file. Or just 'tutorial' to download tutorials."
    )
    parser.add_argument(
        "--config",
        type = str,
        default = "autoprof",
        choices = ["autoprof", "galfit"],
        metavar = "format",
        help = "The type of configuration file being being provided. One of: autoprof, galfit.",
    )
    parser.add_argument(
        "-v", "--version",
        action = "version",
        version = f"%(prog)s {__version__}",
        help = "print the current AutoProf version to screen",
    )
    parser.add_argument(
        "--log",
        type = str,
        metavar = "logfile.log",
        help = "set the log file name for AutoProf. use 'none' to suppress the log file.",
    )
    parser.add_argument(
        "-q",
        action = "store_true",
        help = "quiet flag to stop command line output, only print to log file",
    )
    parser.add_argument(
        "--dtype",
        type = str,
        choices = ["float64", "float32"],
        metavar = "datatype",
        help = "set the float point precision. Must be one of: float64, float32",
    )
    parser.add_argument(
        "--device",
        type = str,
        choices = ["cpu", "gpu"],
        metavar = "device",
        help = "set the device for AutoProf to use for computations. Must be one of: cpu, gpu",
    )
    
    args = parser.parse_args()
    
    if args.log is not None:
        AP_config.set_logging_output(stdout = not args.q, filename = None if args.log == "none" else args.log)
    elif args.q:
        AP_config.set_logging_output(stdout = not args.q, filename = "AutoProf.log")

    if args.dtype is not None:
        AP_config.dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device is not None:
        AP_config.device = "cpu" if args.device == "cpu" else "cuda:0"

    if args.filename is None:
        raise RuntimeError("Please pass a config file to autoprof. See 'autoprof --help' for more information, or go to https://connorstoneastro.github.io/AutoProf/getting_started.html")
    if args.filename in ["tutorial", "tutorials"]:
        import requests
        tutorials = [
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/GettingStarted.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/GroupModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/ModelZoo.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/JointModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/CustomModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf/main/docs/tutorials/simple_config.py",
        ]
        for url in tutorials:
            try:
                R = requests.get(url)
                with open(url[url.rfind("/")+1:], "w") as f:
                    f.write(R.text)
            except:
                print(f"WARNING: couldn't find tutorial: {url[url.rfind('/')+1:]} check internet conection")
                    
        AP_config.ap_logger.info("collected the tutorials")
    elif args.config == "autoprof":
        from .parse_config import basic_config
        basic_config(args.filename)
    elif args.config == "galfit":
        from .parse_config import galfit_config
        galfit_config(args.filename)
    else:
        raise ValueError(f"Unrecognized configuration file format {args.config}. Should be one of: autoprof, galfit")
