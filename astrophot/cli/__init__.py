"""The ``astrophot`` command.

    ~$ astrophot single target.fits [OPTIONS]
    ~$ astrophot segmap target.fits segmap.fits [OPTIONS]

Each sub-command is a self contained script in this directory, so that it can
also be copied out with ``--dump`` and edited.
"""

import sys
from importlib import import_module

__all__ = ("main",)

# sub-command: (module with a ``main(argv, prog)`` function, one line description)
COMMANDS = {
    "single": ("cli_single", "fit one model, plus a sky level, to one image"),
    "segmap": ("cli_segmap", "fit every object of a segmentation map"),
}


def usage():
    lines = ["usage: astrophot <command> [options]", "", "commands:"]
    lines += [f"  {name:<8} {description}" for name, (_, description) in COMMANDS.items()]
    lines += [
        "",
        "options:",
        "  -h, --help     show this message",
        "  -V, --version  show the AstroPhot version",
    ]
    lines += ["", "Use `astrophot <command> --help` for the options of a command."]
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return
    if argv[0] in ("-V", "--version"):
        from .. import __version__

        print(f"astrophot {__version__}")
        return
    command = argv[0]
    if command not in COMMANDS:
        sys.exit(f"error: unknown command '{command}'\n\n{usage()}")
    module = import_module(f".{COMMANDS[command][0]}", __name__)
    module.main(argv[1:], prog=f"astrophot {command}")
