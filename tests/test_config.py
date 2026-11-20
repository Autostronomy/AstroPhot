"""Logging of AstroPhot messages (astrophot/config.py).

The `default_logger` fixture of conftest.py puts the logger back after each test.
"""

import logging
import subprocess
import sys

import astrophot as ap
import pytest


def test_import_leaves_logging_alone(tmp_path):
    """Importing prints to the screen only, and does not touch the logging of anyone else."""
    code = """
import logging, os
import astrophot as ap
root = logging.getLogger()
assert root.handlers == [] and root.level == logging.WARNING, "root logger was changed"
assert ap.config.logger is logging.getLogger("astrophot")
ap.config.logger.info("printed message")
logging.getLogger("another.package").info("not for astrophot")
assert os.listdir() == [], "no log file until set_logger is called"
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "printed message" in result.stdout
    assert "not for astrophot" not in result.stdout + result.stderr


def test_set_logger_file_and_screen(tmp_path, capsys):
    logger = ap.set_logger()
    assert logger is ap.config.logger
    logger.info("to both")
    logger.warning("careful")
    assert capsys.readouterr().out == "to both\nWARNING: careful\n"
    log = (tmp_path / "AstroPhot.log").read_text().splitlines()
    assert len(log) == 2 and log[0].endswith(":INFO: to both"), "file has the time and level"

    # every call replaces the outputs of the call before it
    ap.set_logger("other.log", stdout=False)
    logger.info("file only")
    assert capsys.readouterr().out == ""
    assert "file only" in (tmp_path / "other.log").read_text()
    assert "file only" not in (tmp_path / "AstroPhot.log").read_text()

    # added to the end of an existing file, unless asked to start again
    ap.set_logger(filename=None)
    ap.set_logger("other.log", stdout=False, mode="w", file_format="%(levelname)s|%(message)s")
    logger.info("fresh")
    assert (tmp_path / "other.log").read_text() == "INFO|fresh\n"


def test_set_logger_silent(capsys):
    ap.set_logger(filename=None, stdout=False)
    ap.config.logger.warning("nobody hears this")
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_set_logger_levels(tmp_path, capsys):
    logger = ap.set_logger(tmp_path / "all.log", level="DEBUG", stdout_level=logging.WARNING)
    logger.debug("detail")
    logger.info("progress")
    logger.warning("careful")
    assert capsys.readouterr().out == "WARNING: careful\n"
    assert len((tmp_path / "all.log").read_text().splitlines()) == 3

    with pytest.raises(ValueError):
        ap.set_logger(tmp_path / "bad.log", level="LOUD")
    logger.warning("still set up")
    assert capsys.readouterr().out == "WARNING: still set up\n", "a bad call changes nothing"
    assert len((tmp_path / "all.log").read_text().splitlines()) == 4


class Recorder(logging.Handler):
    """Handler like one a user would add, it keeps the messages it is given."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(f"{record.name}: {record.getMessage()}")


def test_set_logger_plays_nice(capsys):
    """Handlers of the user are kept, and messages only reach the root logger when asked."""
    root = logging.getLogger()
    on_root, on_astrophot = Recorder(), Recorder()
    root_before = list(root.handlers)
    root.addHandler(on_root)
    ap.config.logger.addHandler(on_astrophot)
    try:
        ap.set_logger(filename=None)
        ap.set_logger(filename=None)
        ap.config.logger.info("once")
        assert capsys.readouterr().out == "once\n", "repeated calls should not repeat the output"
        assert on_astrophot.messages == ["astrophot: once"], "handler of the user should be kept"
        assert on_root.messages == [], "nothing goes to the root logger by default"

        ap.set_logger(filename=None, stdout=False, propagate=True)
        ap.config.logger.info("handed over")
        assert capsys.readouterr().out == ""
        assert on_root.messages == ["astrophot: handed over"]
        assert root.handlers == root_before + [on_root], "root logger should not be changed"
    finally:
        root.removeHandler(on_root)
        ap.config.logger.removeHandler(on_astrophot)


def test_set_logging_output_deprecated(tmp_path, capsys):
    with pytest.warns(DeprecationWarning):
        ap.config.set_logging_output(stdout=False, filename=tmp_path / "old.log")
    ap.config.logger.info("old name")
    assert capsys.readouterr().out == ""
    assert "old name" in (tmp_path / "old.log").read_text()
