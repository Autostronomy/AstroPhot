import numpy
from astropy.io import fits
from ..image import Target_Image
from ..models import AutoPhot_Model
from ..fit import LM

__all__ = ["galfit_config"]

galfit_object_type_map = {
    "sersic": "sersic galaxy model",
    "sky": "flat sky model",
}

galfit_parameter_map = {
    "sersic galaxy model": {
        "1": ["centerpix", 2],
        "3": ["totalmag", 1],
        "4": ["Repix", 1],
        "5": ["n", 1],
        "9": ["q", 1],
        "10": ["PAdeg", 1],
    }
}


def space_split(l):
    items = list(ls.strip() for ls in l.split(" "))
    index = 0
    while index < len(items):
        if items[index] == "":
            items.pop(index)
        else:
            index += 1
    return items


def galfit_config(config_file):
    if True:
        raise NotImplementedError(
            "galfit configuration file interface under construction"
        )
    with open(config_file, "r") as f:
        config_lines = f.readlines()
    # Header info
    headerinfo = {}
    for line in config_lines:
        # remove comment from line and strip whitespace
        comment = line.find("#")
        if comment >= 0:
            line = line[:comment].strip()
        if line == "":
            continue
        if line.startswith("A)"):
            headerinfo["target_file"] = line[2:].strip()
        if line.startswith("B)"):
            headerinfo["saveto_model"] = line[2:].strip()
        if line.startswith("C)"):
            headerinfo["varaince_file"] = line[2:].strip()
        if line.startswith("D)"):
            headerinfo["psf_file"] = line[2:].strip()
        if line.startswith("E)"):
            headerinfo["psf_upample"] = line[2:].strip()
        if line.startswith("F)"):
            headerinfo["mask_file"] = line[2:].strip()
        if line.startswith("G)"):
            headerinfo["constraints_file"] = line[2:].strip()
        if line.startswith("H)"):
            headerinfo["fit_window"] = line[2:].strip()
        if line.startswith("I)"):
            headerinfo["convolution_window"] = line[2:].strip()
        if line.startswith("J)"):
            headerinfo["target_zeropoint"] = line[2:].strip()
        if line.startswith("K)"):
            headerinfo["target_pixelscale"] = line[2:].strip()

    # Object info
    objects = []
    in_object = False
    for line in config_lines:
        # remove comment from line and strip whitespace
        comment = line.find("#")
        if comment >= 0:
            linem = line[:comment].strip()
        if linem == "":
            continue

        # New model added to the fit
        if linem.startswith("0)"):
            objects.append({"model_type": galfit_object_type_map[linem[2:].strip()]})
            in_object = True
        # Model finished adding
        if linem.startswith("Z)"):
            in_object = False

        # Collect the parameters
        if in_object:
            param = linem[: linem.find(")")]
            objects[-1][
                galfit_parameter_map[objects[-1]["model_type"]][param][0]
            ] = space_split(linem[linem.find(")") + 1 :])
            if len(
                objects[-1][galfit_parameter_map[objects[-1]["model_type"]][param][0]]
            ) != (2 * galfit_parameter_map[objects[-1]["model_type"]][param][1]):
                raise ValueError(
                    f"Incorrectly formatted line in GALFIT config file:\n{line}"
                )

    # Format parameters
    for i in range(len(objects)):
        autophot_object = {
            "model_type": objects[i]["model_type"],
        }

        # common params
        if "centerpix" in objects[i]:
            autophot_object["center"] = {
                "value": [
                    float(objects[i]["centerpix"][0]) * headerinfo["target_pixelscale"],
                    float(objects[i]["centerpix"][1]) * headerinfo["target_pixelscale"],
                ],
                "locked": bool(objects[i]["centerpix"][2]),
            }
        if "Repix" in objects[i]:
            autophot_object["Re"] = {
                "value": float(objects[i]["Repix"][0])
                * headerinfo["target_pixelscale"],
                "locked": bool(objects[i]["Repix"][1]),
            }
        if "q" in objects[i]:
            autophot_object["q"] = {
                "value": float(objects[i]["q"][0]),
                "locked": bool(objects[i]["q"][1]),
            }
        if "PAdeg" in objects[i]:
            autophot_object["PA"] = {
                "value": float(objects[i]["PAdeg"][0]) * np.pi / 180,
                "locked": bool(objects[i]["PAdeg"][1]),
            }
