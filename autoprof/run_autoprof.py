#!/usr/bin/python3

import shutil
import sys
import os
import importlib
from autoprof.pipeline import build_pipeline
from autoprof.state import State
from flow import Pipe

def GetOptions(c):
    """
    Extract all of the AutoProf user optionional parameters form the config file.
    User options are identified as any python object that starts with "ap\_" in the
    variable name.
    """
    newoptions = {}
    for var in dir(c):
        if var.startswith("ap_"):
            val = getattr(c, var)
            if not val is None:
                newoptions[var] = val

    return newoptions

if __name__ == "__main__":
    py3 = shutil.which("python3").strip()
    with open(__file__, "r") as f:
        raw = f.readlines()
    if py3 != raw[0][2:].strip():
        with open(__file__, "w") as f:
            raw[0] = "#!" + py3 + "\n"
            f.writelines(raw)
        print(
            "Encountered a minor hiccup locating python3 and fixed it. Please just run again and everything should work.\n If you encounter further issues such as 'numpy not found' even though numpy is installed, see the trouble shooting guide in the installation page of the documentation."
        )
        
    assert (
        len(sys.argv) >= 2
    ), "Please supply a config file to AutoProf."

    config_file = sys.argv[1]

    # Import the config file regardless of where it is from
    if '/' in config_file:
        startat = config_file.rfind('/')+1
    else:
        startat = 0
    if '.' in config_file:
        use_config = config_file[startat:config_file.rfind('.')]
    else:
        use_config = config_file[startat:]
    if '/' in config_file:
        sys.path.append(config_file[:config_file.rfind('/')])
    try:
        c = importlib.import_module(use_config)
    except:
        sys.path.append(os.getcwd())
        c = importlib.import_module(use_config)
    use_options = GetOptions(c)
    
    AP_State = State(**use_options)
    AP_Chart = build_pipeline(**use_options)
    if hasattr(AP_State, "__next__"):
        print("parallel pipeline")
        AP_Pipe = Pipe(AP_Chart, return_success = True)
        AP_Pipe(AP_State)
    else:
        print("single pipeline")
        AP_Chart(AP_State)
