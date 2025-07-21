import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob
import pytest

notebooks = glob.glob("../docs/source/tutorials/*.ipynb")


@pytest.mark.parametrize("nb_path", notebooks)
def test_notebook_runs(nb_path):
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "./"}})
