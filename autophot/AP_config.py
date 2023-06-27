import sys
import logging
import torch

__all__ = ["ap_dtype", "ap_device", "ap_logger", "set_logging_output"]

ap_dtype = torch.float64
ap_device = "cuda:0" if torch.cuda.is_available() else "cpu"
ap_verbose = 0

logging.basicConfig(
    filename="AutoPhot.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s: %(message)s",
)
ap_logger = logging.getLogger()
out_handler = logging.StreamHandler(sys.stdout)
out_handler.setLevel(logging.INFO)
out_handler.setFormatter(logging.Formatter("%(message)s"))
ap_logger.addHandler(out_handler)


def set_logging_output(stdout=True, filename=None, **kwargs):
    """
    Change the logging system for AutoPhot.
    Here you can set whether output prints to screen or to a logging file.
    This function will remove all handlers from the current logger in ap_logger,
    then add new handlers based on the input to the function.

    Parameters:
        stdout (bool): If True, log messages will be printed to the console (standard output). Default is True.
        filename (str): If given as a string, this will be the name of the file that log messages are written to.
                        If None, no logging file will be used. Default is None.
        stdout_level (logging.INFO): The logging level of messages written to stdout, this can be different from the file level.
                            Default is logging.INFO.
        stdout_formatter (logging.Formatter): A logging.Formatter object which determines what information to include with the logging message only when printing to stdout. Default is logging.Formatter('%(message)s').
        filename_level (int): The logging level of messages written to the log file, this can be different from the stdout level. Default is logging.INFO.
        filename_formatter (logging.Formatter): A logging.Formatter object which determines what information to include with the logging message only when printing to the log file. Default is logging.Formatter('%(asctime)s:%(levelname)s: %(message)s').

    """
    hi = 0
    while hi < len(ap_logger.handlers):
        if isinstance(ap_logger.handlers[hi], logging.StreamHandler):
            ap_logger.removeHandler(ap_logger.handlers[hi])
        elif isinstance(ap_logger.handlers[hi], logging.FileHandler):
            ap_logger.removeHandler(ap_logger.handlers[hi])
        else:
            hi += 1

    if stdout:
        out_handler = logging.StreamHandler(sys.stdout)
        out_handler.setLevel(kwargs.get("stdout_level", logging.INFO))
        out_handler.setFormatter(
            kwargs.get("stdout_formatter", logging.Formatter("%(message)s"))
        )
        ap_logger.addHandler(out_handler)
        ap_logger.debug("logging now going to stdout")
    if filename is not None:
        out_handler = logging.FileHandler(filename)
        out_handler.setLevel(kwargs.get("filename_level", logging.INFO))
        out_handler.setFormatter(
            kwargs.get(
                "filename_formatter",
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"),
            )
        )
        ap_logger.addHandler(out_handler)
        ap_logger.debug("logging now going to %s" % filename)
