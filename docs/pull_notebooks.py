import requests

tutorials = [
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/GettingStarted.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/GroupModels.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/ModelZoo.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/JointModels.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/FittingMethods.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/CustomModels.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/AdvancedPSFModels.ipynb",
    "https://raw.github.com/Autostronomy/AstroPhot-tutorials/main/tutorials/ConstrainedModels.ipynb",
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
