import sys
import os
import importlib
import numpy as np
from astropy.io import fits
from ..image import Target_Image
from ..models import AutoPhot_Model
from ..fit import LM
from .. import AP_config

__all__ = ["basic_config"]


def GetOptions(c):
    newoptions = {}
    for var in dir(c):
        if var.startswith("ap_"):
            val = getattr(c, var)
            if not val is None:
                newoptions[var] = val
    return newoptions


def import_configfile(config_file):
    if "/" in config_file:
        startat = config_file.rfind("/") + 1
    else:
        startat = 0
    if "." in config_file:
        use_config = config_file[startat : config_file.rfind(".")]
    else:
        use_config = config_file[startat:]
    if startat > 0:
        sys.path.append(os.path.abspath(config_file[: config_file.rfind("/")]))
    else:
        sys.path.append(os.getcwd())
    c = importlib.import_module(use_config)
    return c


def basic_config(config_file):
    c = import_configfile(config_file)  # importlib.import_module(config_file)
    config = GetOptions(c)

    # Parse Target
    ######################################################################
    AP_config.ap_logger.info("Collecting target information")
    target = config.get("ap_target", None)
    if target is None:
        target_file = config.get("ap_target_file", None)
        target_hdu = config.get("ap_target_hdu", 0)
        variance_file = config.get("ap_variance_file", None)
        variance_hdu = config.get("ap_variance_hdu", 0)
        target_pixelscale = config.get("ap_target_pixelscale", None)
        target_zeropoint = config.get("ap_target.zeropoint", None)
        target_origin = config.get("ap_target_origin", None)

        if variance_file is not None:
            var_data = np.array(
                fits.open(target_file)[target_hdu].data, dtype=np.float64
            )
        else:
            var_data = None
        if target_file is not None:
            data = np.array(fits.open(target_file)[target_hdu].data, dtype=np.float64)
            target = Target_Image(
                data=data,
                pixelscale=target_pixelscale,
                zeropoint=target_zeropoint,
                variance=var_data,
                origin=target_origin,
            )

    # Parse Models
    ######################################################################
    AP_config.ap_logger.info("Constructing models")
    model_info_list = config.get("ap_models", [])
    name_order = config.get(
        "ap_model_name_order",
        list(n[9:] for n in filter(lambda k: k.startswith("ap_model_"), config.keys())),
    )
    for name in name_order:
        key_name = "ap_model_" + name
        model_info_list.append(config[key_name])
        if "name" not in model_info_list[-1]:
            model_info_list[-1]["name"] = name
    model_list = []
    for model in model_info_list:
        model_list.append(AutoPhot_Model(target=target, **model))

    MODEL = AutoPhot_Model(
        name="AutoPhot",
        model_type="group model",
        models=model_list,
        target=target,
    )

    # Parse Optimize
    ######################################################################
    AP_config.ap_logger.info("Running optimization")
    MODEL.initialize()

    optim_type = config.get("ap_optimizer", "LM")
    optim_kwargs = config.get("ap_optimizer_kwargs", {})
    if optim_type is None:
        # perform no optimization, simply write the autophot model and the requested images
        pass
    elif optim_type == "LM":
        result = LM(MODEL, **optim_kwargs).fit()

    # Parse Save
    ######################################################################
    AP_config.ap_logger.info("Saving model")
    model_save = config.get("ap_saveto_model", "AutoPhot.yaml")
    MODEL.save(model_save)

    model_image_save = config.get("ap_saveto_model_image", None)
    if model_image_save is not None:
        MODEL().save(model_image_save)

    model_residual_save = config.get("ap_saveto_model_residual", None)
    if model_residual_save is not None:
        (target - MODEL()).save(model_residual_save)
