import platform
import glob
import pytest
import runpy
import subprocess
import os
import caskade as ck
import astrophot as ap

pytestmark = pytest.mark.skipif(
    platform.system() in ["Windows", "Darwin"],
    reason="Graphviz not installed on Windows runner",
)


notebooks = glob.glob(
    os.path.join(
        os.path.split(os.path.dirname(__file__))[0], "docs", "source", "tutorials", "*.ipynb"
    )
)


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
    if ap.backend.backend == "jax":
        pytest.skip("Requires torch backend")
    convert_notebook_to_py(nb_path)
    runpy.run_path(nb_path.replace(".ipynb", ".py"), run_name="__main__")
    ck.backend.backend = "torch"
    ap.backend.backend = "torch"
    cleanup_py_scripts(nb_path)
