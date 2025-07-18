import astrophot as ap
import torch
import numpy as np
import torch

import astrophot as ap
from utils import make_basic_sersic, make_basic_gaussian_psf


def test_jointmodel_creation():
    np.random.seed(12345)
    shape = (10, 15)
    tar1 = ap.TargetImage(
        name="target1",
        data=np.random.normal(loc=0, scale=1.4, size=shape),
        pixelscale=0.8,
        variance=np.ones(shape) * (1.4**2),
    )
    shape2 = (33, 42)
    tar2 = ap.TargetImage(
        name="target2",
        data=np.random.normal(loc=0, scale=1.4, size=shape2),
        pixelscale=0.3,
        variance=np.ones(shape2) * (1.4**2),
    )

    tar = ap.TargetImageList([tar1, tar2])

    mod1 = ap.models.FlatSky(
        name="base model 1",
        target=tar1,
    )
    mod2 = ap.models.FlatSky(
        name="base model 2",
        target=tar2,
    )

    smod = ap.Model(
        name="group model",
        model_type="group model",
        models=[mod1, mod2],
        target=tar,
    )

    smod.initialize()
    assert torch.all(torch.isfinite(smod().flatten("data"))).item(), "model_image should be real"

    fm = smod.fit_mask()
    for fmi in fm:
        assert torch.sum(fmi).item() == 0, "this fit_mask should not mask any pixels"


def test_psfgroupmodel_creation():
    tar = make_basic_gaussian_psf()

    mod1 = ap.Model(
        name="base model 1",
        model_type="moffat psf model",
        target=tar,
    )

    mod2 = ap.Model(
        name="base model 2",
        model_type="moffat psf model",
        target=tar,
    )

    smod = ap.Model(
        name="group model",
        model_type="psf group model",
        models=[mod1, mod2],
        target=tar,
    )

    smod.initialize()

    assert torch.all(smod().data >= 0), "PSF group sample should be greater than or equal to zero"
