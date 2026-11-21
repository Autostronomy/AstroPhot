import logging
import sys
import warnings
from pathlib import Path

__all__ = ["DTYPE", "DEVICE", "logger", "set_logger", "set_logging_output"]

DTYPE = None
DEVICE = None

# Every message of AstroPhot goes through this logger. It is an ordinary named
# logger, so it can also be configured with the tools of the `logging` module.
# The root logger, which belongs to the program that imports AstroPhot, is
# never touched.
logger = logging.getLogger("astrophot")
# stops python from printing warnings to stderr when `set_logger` has turned
# every output off
logger.addHandler(logging.NullHandler())

STDOUT_FORMAT = "%(message)s"
FILE_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"


class _StdoutHandler(logging.StreamHandler):
    """Writes to `sys.stdout` as it is when the message is written.

    A plain StreamHandler keeps the `sys.stdout` it was created with, and
    would miss a notebook, `contextlib.redirect_stdout` or pytest replacing it.
    """

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stdout


class _ScreenFormatter(logging.Formatter):
    """The message alone, with the level in front for warnings and errors."""

    def format(self, record):
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname}: {message}"
        return message


def _formatter(fmt, default):
    if fmt is None:
        return default
    return fmt if isinstance(fmt, logging.Formatter) else logging.Formatter(fmt)


def _level(level):
    """The number of a logging level, given as that number or as its name."""
    if isinstance(level, int):
        return level
    names = logging.getLevelNamesMapping()
    if level not in names:
        raise ValueError(f"Unknown level: {level!r}")
    return names[level]


def _header():
    """What was running when a log file was started, for its first line."""
    from importlib.metadata import version

    from . import __version__
    from .backend_obj import backend

    header = f"AstroPhot {__version__}, {backend.backend} {version(backend.backend)} backend"
    if DTYPE is not None:
        header += f", dtype {DTYPE}, device {DEVICE}"
    return header


def set_logger(
    filename="AstroPhot.log",
    stdout=True,
    level="INFO",
    *,
    stdout_level=None,
    file_level=None,
    stdout_format=None,
    file_format=None,
    mode="a",
    propagate=False,
):
    """Choose where the messages of AstroPhot go.

    When AstroPhot is imported its messages are only printed to the screen.
    ``ap.set_logger()`` also starts writing them to the file ``AstroPhot.log``
    in the current directory. Every call replaces the outputs set by the call
    before it, so ``ap.set_logger(filename=None)`` goes back to the screen
    alone and ``ap.set_logger(filename=None, stdout=False)`` is silent.

    A log file starts with a line giving the version of AstroPhot, the backend
    and, with PyTorch, the dtype and device in use.

    Only the "astrophot" logger is changed, and only the handlers that this
    function made. Logging set up by your own program, by other packages, or
    handlers that you added to ``ap.config.logger`` yourself are left alone.

    Parameters:
        filename (str or Path): File to write the messages to, None for no
            file. Default is "AstroPhot.log".
        stdout (bool): Print the messages to the screen. Default is True.
        level (str or int): Messages below this level are dropped, one of
            "DEBUG", "INFO", "WARNING", "ERROR" or the matching constant of
            the `logging` module. The fitters report their progress at "INFO"
            and extra detail at "DEBUG", so "WARNING" hides the progress of
            every fit and "DEBUG" shows the most. Default is "INFO".
        stdout_level (str or int): Level for the screen alone, it can not be
            lower than `level`. Default is the same as `level`.
        file_level (str or int): Level for the file alone, it can not be lower
            than `level`. Default is the same as `level`.
        stdout_format (str or logging.Formatter): Format of the messages on the
            screen. Default is the message alone, with "WARNING: " or
            "ERROR: " in front for those levels.
        file_format (str or logging.Formatter): Format of the messages in the
            file. Default is "%(asctime)s:%(levelname)s: %(message)s".
        mode (str): "a" to add to the end of an existing file, "w" to start it
            again. Default is "a".
        propagate (bool): Also pass the messages on to the root logger, so that
            they reach the handlers that your own program has set up with the
            `logging` module. Use this with ``filename=None, stdout=False`` to
            leave the output entirely to your program. Default is False.

    Returns:
        The "astrophot" logger, which is also ``ap.config.logger``.
    """
    # Check every argument before anything is opened or changed, so that a bad
    # one leaves the logger and an existing log file as they were
    level = _level(level)
    stdout_level = level if stdout_level is None else _level(stdout_level)
    file_level = level if file_level is None else _level(file_level)
    stdout_format = _formatter(stdout_format, _ScreenFormatter(STDOUT_FORMAT))
    file_format = _formatter(file_format, logging.Formatter(FILE_FORMAT))

    handlers = []
    if filename is not None:
        # the one step that can still fail, so it comes before the other handler
        handler = logging.FileHandler(Path(filename).expanduser(), mode=mode, encoding="utf-8")
        handler.setFormatter(file_format)
        handler.setLevel(file_level)
        handlers.append(handler)
    if stdout:
        handler = _StdoutHandler()
        handler.setFormatter(stdout_format)
        handler.setLevel(stdout_level)
        handlers.insert(0, handler)
    logger.setLevel(level)
    logger.propagate = propagate

    for handler in list(logger.handlers):
        if getattr(handler, "_astrophot_owned", False):
            logger.removeHandler(handler)
            handler.close()
    for handler in handlers:
        handler._astrophot_owned = True
        logger.addHandler(handler)
        if isinstance(handler, logging.FileHandler):
            # written whatever the level of the file
            handler.emit(logger.makeRecord(logger.name, logging.INFO, "", 0, _header(), (), None))
    return logger


def set_logging_output(stdout=True, filename=None, **kwargs):
    """Deprecated, use `set_logger`."""
    warnings.warn(
        "set_logging_output is deprecated, use ap.set_logger instead",
        DeprecationWarning,
        2,
    )
    renamed = {"filename_level": "file_level", "stdout_formatter": "stdout_format"}
    renamed["filename_formatter"] = "file_format"
    kwargs = {renamed.get(key, key): value for key, value in kwargs.items()}
    return set_logger(filename=filename, stdout=stdout, **kwargs)


# print to the screen only, until the user asks for more
set_logger(filename=None)
