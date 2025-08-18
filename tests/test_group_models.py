import astrophot as ap
import numpy as np

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
    assert ap.backend.all(
        ap.backend.isfinite(smod().flatten("data"))
    ).item(), "model_image should be real"

    fm = smod.fit_mask()
    for fmi in fm:
        assert ap.backend.sum(fmi).item() == 0, "this fit_mask should not mask any pixels"


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

    assert ap.backend.all(
        smod().data >= 0
    ), "PSF group sample should be greater than or equal to zero"


def test_joint_multi_band_multi_object():
    target1 = make_basic_sersic(52, 53, name="target1")
    target2 = make_basic_sersic(48, 65, name="target2")
    target3 = make_basic_sersic(60, 49, name="target3")
    target4 = make_basic_sersic(60, 49, name="target4")

    # fmt: off
    model11 = ap.Model(name="model11", model_type="sersic galaxy model", window=(0, 50, 5, 52), target=target1)
    model12 = ap.Model(name="model12", model_type="sersic galaxy model", window=(3, 53, 0, 49), target=target1)
    model1 = ap.Model(name="model1", model_type="group model", models=[model11, model12], target=target1)

    model21 = ap.Model(name="model21", model_type="sersic galaxy model", window=(1, 62, 10, 48), target=target2)
    model22 = ap.Model(name="model22", model_type="sersic galaxy model", window=(2, 60, 5, 49), target=target2)
    model2 = ap.Model(name="model2", model_type="group model", models=[model21, model22], target=target2)

    model31 = ap.Model(name="model31", model_type="sersic galaxy model", window=(1, 62, 10, 48), target=target3)
    model32 = ap.Model(name="model32", model_type="sersic galaxy model", window=(2, 60, 5, 49), target=target3)
    model3 = ap.Model(name="model3", model_type="group model", models=[model31, model32], target=target3)

    model4 = ap.Model(name="model4", model_type="sersic galaxy model", window=(0, 53, 0, 52), target=target1)

    model51 = ap.Model(name="model51", model_type="sersic galaxy model", window=(0, 65, 0, 48), target=target2)
    model52 = ap.Model(name="model52", model_type="sersic galaxy model", window=(0, 49, 0, 60), target=target3)
    model5 = ap.Model(name="model5", model_type="group model", models=[model51, model52], target=ap.TargetImageList([target2, target3]))

    model = ap.Model(name="joint model", model_type="group model", models=[model1, model2, model3, model4, model5], target=ap.TargetImageList([target1, target2, target3, target4]))
    # fmt: on

    model.initialize()
    mask = model.fit_mask()
    assert len(mask) == 4, "There should be 4 fit masks for the 4 targets"
    for m in mask:
        assert ap.backend.all(ap.backend.isfinite(m)), "this fit_mask should be finite"
    sample = model.sample(window=ap.WindowList([target1.window, target2.window, target3.window]))
    assert isinstance(sample, ap.ImageList), "Sample should be an ImageList"
    for image in sample:
        assert ap.backend.all(ap.backend.isfinite(image.data)), "Sample image data should be finite"
        assert ap.backend.all(image.data >= 0), "Sample image data should be non-negative"

    jacobian = model.jacobian()
    assert isinstance(jacobian, ap.ImageList), "Jacobian should be an ImageList"
    for image in jacobian:
        assert ap.backend.all(
            ap.backend.isfinite(image.data)
        ), "Jacobian image data should be finite"

    window = model.window
    assert isinstance(window, ap.WindowList), "Window should be a WindowList"
