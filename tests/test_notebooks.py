"""Run every tutorial notebook of docs/source/tutorials from start to end.

These check that the tutorials still work, they are by far the slowest tests.
Leave them out with ``pytest -m "not notebooks"`` or pick one with
``pytest -k FittingMethods``.
"""

import importlib.util
import platform
import runpy
import shutil
import subprocess
from pathlib import Path

import astrophot as ap
import caskade as ck
import pytest

pytestmark = [
    pytest.mark.notebooks,
    pytest.mark.skipif(
        platform.system() in ["Windows", "Darwin"],
        reason="Graphviz not installed on Windows runner",
    ),
]

TUTORIALS = Path(__file__).resolve().parent.parent / "docs" / "source" / "tutorials"


def convert_notebook_to_py(notebook, directory):
    """Write the code of the notebook as a script in `directory`, and return its path."""
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "python", "--output-dir", str(directory), str(notebook)],
        check=True,
    )
    script = directory / (notebook.stem + ".py")
    lines = script.read_text().splitlines(keepends=True)
    # Remove get_ipython() lines to avoid errors in script execution
    lines = [line for line in lines if not line.startswith("get_ipython()")]
    # downloads in the tutorials should fail rather than hang
    script.write_text("import socket\nsocket.setdefaulttimeout(120)\n" + "".join(lines))
    return script


@pytest.mark.parametrize("notebook", sorted(TUTORIALS.glob("*.ipynb")), ids=lambda nb: nb.stem)
def test_notebook(notebook, tmp_path):
    if ap.backend.backend != "torch":
        pytest.skip("Requires torch backend")
    if importlib.util.find_spec("caustics") is None and "GravitationalLensing" in notebook.name:
        pytest.skip("This tutorial requires caustics")

    script = convert_notebook_to_py(notebook, tmp_path)
    # The tutorials read their images from the working directory, which is tmp_path
    for target in TUTORIALS.glob("*target_image*.fits"):
        shutil.copy(target, tmp_path)
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        # some tutorials switch to the JAX backend
        ck.backend.backend = "torch"
        ap.backend.backend = "torch"
