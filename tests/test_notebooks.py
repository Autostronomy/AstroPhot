import platform
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob
import pytest
import runpy
import subprocess
import os

pytestmark = pytest.mark.skipif(
    platform.system() in ["Windows", "Darwin"],
    reason="Graphviz not installed on Windows runner",
)

notebooks = glob.glob("../docs/source/tutorials/*.ipynb")


# @pytest.mark.parametrize("nb_path", notebooks)
# def test_notebook_runs(nb_path):
#     with open(nb_path) as f:
#         nb = nbformat.read(f, as_version=4)
#     ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
#     ep.preprocess(nb, {"metadata": {"path": "./"}})
def convert_notebook_to_py(nbpath):
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "python", nbpath],
        check=True,
    )
    pypath = nbpath.replace(".ipynb", ".py")
    with open(pypath, "r") as f:
        content = f.readlines()
    with open(pypath, "w") as f:
        for line in content:
            if line.startswith("get_ipython()"):
                # Remove get_ipython() lines to avoid errors in script execution
                continue
            f.write(line)


def cleanup_py_scripts(nbpath):
    try:
        os.remove(nbpath.replace(".ipynb", ".py"))
        os.remove(nbpath.replace(".ipynb", ".pyc"))
    except FileNotFoundError:
        pass


@pytest.mark.parametrize("nb_path", notebooks)
def test_notebook(nb_path):
    convert_notebook_to_py(nb_path)
    runpy.run_path(nb_path.replace(".ipynb", ".py"), run_name="__main__")
    cleanup_py_scripts(nb_path)
