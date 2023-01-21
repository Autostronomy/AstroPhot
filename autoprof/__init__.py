from . import models
from . import image
from . import plots
from . import utils
from . import fit
from . import AP_config

# meta data
__version__ = "0.3.0"
__author__ = "Connor Stone"
__email__ = "connorston628@gmail.com"

def run_from_terminal():
    import sys
    import argparse
    AP_config.ap_logger.debug("running from the terminal, not sure if it will catch me.")
    parser = argparse.ArgumentParser(
        prog = "autoprof",
        description = "Fast and flexible photometry reduction package",
        epilog = "Please contact connor stone (connorstone628@gmail.com) for further assistance."
    )

    parser.add_argument("filename", type = str, help = "the path to the configuration file. Or just 'tutorial' to download tutorials.")
    parser.add_argument(
        "-c", "--config",
        type = str,
        default = "autoprof",
        choices = ["autoprof", "galfit"],
        help = "The type of configuration file being being provided. One of: autoprof, galfit.",
    )
    args = parser.parse_args()
    
    if args.filename in ["tutorial", "tutorials"]:
        import requests
        tutorials = [
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/GettingStarted.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/GroupModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/ModelZoo.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/MultibandModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/CustomModels.ipynb",
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
