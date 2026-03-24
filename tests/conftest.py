import matplotlib
import matplotlib.pyplot as plt
import pytest
import numpy as np
import astrophot as ap


@pytest.fixture(autouse=True)
def no_block_show(monkeypatch):
    def close_show(*args, **kwargs):
        # plt.savefig("/dev/null")  # or do nothing
        plt.close("all")

    monkeypatch.setattr(plt, "show", close_show)

    # Also ensure we are in a non-GUI backend
    matplotlib.use("Agg")


@pytest.fixture()
def sersic(request):

    if not hasattr(request, "param"):
        request.param = {}
    np.random.seed(request.param.get("seed", 12345))
    shape = request.param.get("shape", (52, 50))
    mask = request.param.get("mask", None)
    if mask is None:
        mask = np.zeros(shape, dtype=bool)
        mask[0][0] = True
    target = request.param.get("target", None)
    pixelscale = 0.8
    if target is None:
        target = ap.TargetImage(
            data=np.zeros(shape),
            pixelscale=pixelscale,
            psf=ap.utils.initialize.gaussian_psf(2 / pixelscale, 11, pixelscale),
            mask=mask,
            zeropoint=21.5,
        )

    MODEL = ap.models.SersicGalaxy(
        name="basic_sersic_model",
        target=target,
        center=request.param.get("center", [20.5, 21.4]),
        PA=request.param.get("PA", 45 * np.pi / 180),
        q=request.param.get("q", 0.7),
        n=request.param.get("n", 1.5),
        Re=request.param.get("Re", 15.1),
        Ie=request.param.get("Ie", 10.0),
        sampling_mode="quad:5",
    )

    if request.param.get("target", None) is None:
        img = ap.backend.to_numpy(MODEL().data)
        target.data = (
            img
            + np.random.normal(scale=0.5, size=img.shape)
            + np.random.normal(scale=np.sqrt(img) / 10)
        )
        target.variance = 0.5**2 + img / 100

    return MODEL
