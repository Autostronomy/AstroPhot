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
    AP_config.ap_logger.debug("running from the terminal, not sure if it will catch me.")
    if any("tutorial" in A for A in sys.argv):
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
