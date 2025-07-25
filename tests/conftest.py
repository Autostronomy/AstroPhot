import matplotlib
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def no_block_show(monkeypatch):
    def close_show(*args, **kwargs):
        # plt.savefig("/dev/null")  # or do nothing
        plt.close("all")

    monkeypatch.setattr(plt, "show", close_show)

    # Also ensure we are in a non-GUI backend
    matplotlib.use("Agg")
