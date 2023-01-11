from . import models
from . import image
from . import plots
from . import utils
from . import fit

# meta data
__version__ = "0.3.0"
__author__ = "Connor Stone"
__email__ = "connorston628@gmail.com"

def run_from_terminal():
    import sys
    print("running from the terminal, not sure if it will catch me.")
    if any("tutorial" in A for A in sys.argv):
        import requests
        tutorials = [
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/GettingStarted.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/GroupModels.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/ModelZoo.ipynb",
            "https://raw.github.com/ConnorStoneAstro/AutoProf-2/main/docs/tutorials/MultibandModels.ipynb",
        ]
        for url in tutorials:
            R = requests.get(url)
            with open(url[url.rfind("/")+1:], "w") as f:
                f.write(R.text)
        print("collected the tutorials")
