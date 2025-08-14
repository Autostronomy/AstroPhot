import astrophot as ap
import torch
import numpy as np

import pytest

######################################################################
# Image Objects
######################################################################


@pytest.fixture()
def cmos_target():
    arr = ap.backend.zeros((10, 15))
    return ap.CMOSTargetImage(
        data=arr,
        pixelscale=0.7,
        zeropoint=1.0,
        variance=ap.backend.ones_like(arr),
        mask=ap.backend.zeros_like(arr),
        subpixel_loc=(-0.25, -0.25),
        subpixel_scale=0.5,
    )


def test_cmos_image_creation(cmos_target):
    cmos_copy = cmos_target.copy()
    assert cmos_copy.pixelscale == 0.7, "image should track pixelscale"
    assert cmos_copy.zeropoint == 1.0, "image should track zeropoint"
    assert cmos_copy.crpix[0] == 0, "image should track crpix"
    assert cmos_copy.crpix[1] == 0, "image should track crpix"
    assert cmos_copy.subpixel_loc == (-0.25, -0.25), "image should track subpixel location"
    assert cmos_copy.subpixel_scale == 0.5, "image should track subpixel scale"

    print(cmos_target.data.shape)
    i, j = cmos_target.pixel_center_meshgrid()
    assert i.shape == (15, 10), "meshgrid should have correct shape"
    assert j.shape == (15, 10), "meshgrid should have correct shape"

    x, y = cmos_target.coordinate_center_meshgrid()
    assert x.shape == (15, 10), "coordinate meshgrid should have correct shape"
    assert y.shape == (15, 10), "coordinate meshgrid should have correct shape"


def test_cmos_model_sample(cmos_target):
    model = ap.Model(
        name="test cmos",
        model_type="sersic galaxy model",
        target=cmos_target,
        center=(3, 5),
        q=0.7,
        PA=np.pi / 3,
        n=2.5,
        Re=4,
        Ie=1.0,
        sampling_mode="midpoint",
        integrate_mode="bright",
    )
    model.initialize()
    img = model.sample()

    assert isinstance(img, ap.CMOSModelImage), "sampled image should be a CMOSModelImage"
    assert img.pixelscale == cmos_target.pixelscale, "sampled image should have the same pixelscale"
    assert img.zeropoint == cmos_target.zeropoint, "sampled image should have the same zeropoint"
    assert (
        img.subpixel_loc == cmos_target.subpixel_loc
    ), "sampled image should have the same subpixel location"


def test_cmos_image_save_load(cmos_target):
    # Save the image
    cmos_target.save("cmos_image.fits")

    # Load the image
    loaded_image = ap.CMOSTargetImage(filename="cmos_image.fits")

    # Check if the loaded image matches the original
    assert ap.backend.allclose(
        cmos_target.data, loaded_image.data
    ), "Loaded image data should match original"
    assert ap.backend.allclose(
        cmos_target.pixelscale, loaded_image.pixelscale
    ), "Loaded image pixelscale should match original"
    assert ap.backend.allclose(
        cmos_target.zeropoint, loaded_image.zeropoint
    ), "Loaded image zeropoint should match original"
    assert np.allclose(
        cmos_target.subpixel_loc, loaded_image.subpixel_loc
    ), "Loaded image subpixel location should match original"
    assert np.allclose(
        cmos_target.subpixel_scale, loaded_image.subpixel_scale
    ), "Loaded image subpixel scale should match original"
