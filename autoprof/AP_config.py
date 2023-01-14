import torch
import logging
import sys

__all__ = ["ap_dtype", "ap_device", "ap_logger", "set_logging_stdout"]

ap_dtype = torch.float64
ap_device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.basicConfig(filename = "AutoProf.log", level = logging.INFO, format = "%(asctime)s:%(levelname)s: %(message)s")
ap_logger = logging.getLogger()

def set_logging_stdout():
    out_handler = logging.StreamHandler(sys.stdout)
    ap_logger.addHandler(out_handler)
