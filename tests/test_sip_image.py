import astrophot as ap
import numpy as np

import pytest

######################################################################
# Image Objects
######################################################################


@pytest.fixture()
def sip_target():
    arr = np.zeros((10, 15))
    return ap.SIPTargetImage(
        data=arr,
        pixelscale=1.0,
        zeropoint=1.0,
        variance=np.ones_like(arr),
        mask=np.zeros_like(arr),
        sipA={(1, 0): 1e-4, (0, 1): 1e-4, (2, 3): -1e-5},
        sipB={(1, 0): -1e-4, (0, 1): 5e-5, (2, 3): 2e-6},
        # sipAP={(1, 0): -1e-4, (0, 1): -1e-4, (2, 3): 1e-5},
        # sipBP={(1, 0): 1e-4, (0, 1): -5e-5, (2, 3): -2e-6},
    )


def test_sip_image_creation(sip_target):
    assert sip_target.pixelscale == 1.0, "image should track pixelscale"
    assert sip_target.zeropoint == 1.0, "image should track zeropoint"
    assert sip_target.crpix[0] == 0, "image should track crpix"
    assert sip_target.crpix[1] == 0, "image should track crpix"

    slicer = ap.Window((7, 13, 4, 7), sip_target)
    sliced_image = sip_target[slicer]
    assert sliced_image.crpix[0] == -7, "crpix of subimage should give relative position"
    assert sliced_image.crpix[1] == -4, "crpix of subimage should give relative position"
    assert sliced_image.shape == (6, 3), "sliced image should have correct shape"
    assert sliced_image.pixel_area_map.shape == (
        6,
        3,
    ), "sliced image should have correct pixel area map shape"
    assert sliced_image.distortion_ij.shape == (
        2,
        6,
        3,
    ), "sliced image should have correct distortion shape"
    assert sliced_image.distortion_IJ.shape == (
        2,
        6,
        3,
    ), "sliced image should have correct distortion shape"

    sip_model_image = sip_target.model_image(upsample=2, pad=1)
    assert sip_model_image.shape == (32, 22), "model image should have correct shape"
    assert sip_model_image.pixel_area_map.shape == (
        32,
        22,
    ), "model image pixel area map should have correct shape"
    assert sip_model_image.distortion_ij.shape == (
        2,
        32,
        22,
    ), "model image distortion model should have correct shape"
    assert sip_model_image.distortion_IJ.shape == (
        2,
        32,
        22,
    ), "model image distortion model should have correct shape"

    # reduce
    sip_model_reduce = sip_model_image.reduce(scale=1)
    assert sip_model_reduce is sip_model_image, "reduce should return the same image if scale is 1"
    sip_model_reduce = sip_model_image.reduce(scale=2)
    assert sip_model_reduce.shape == (16, 11), "reduced model image should have correct shape"

    # crop
    sip_model_crop = sip_model_image.crop(1)
    assert sip_model_crop.shape == (30, 20), "cropped model image should have correct shape"
    sip_model_crop = sip_model_image.crop([1])
    assert sip_model_crop.shape == (30, 20), "cropped model image should have correct shape"
    sip_model_crop = sip_model_image.crop([1, 2])
    assert sip_model_crop.shape == (30, 18), "cropped model image should have correct shape"
    sip_model_crop = sip_model_image.crop([1, 2, 3, 4])
    assert sip_model_crop.shape == (29, 15), "cropped model image should have correct shape"

    sip_model_crop.fluxdensity_to_flux()
    assert ap.backend.all(
        sip_model_crop.data >= 0
    ), "cropped model image data should be non-negative after flux density to flux conversion"


def test_sip_image_wcs_roundtrip(sip_target):
    """
    Test that the WCS roundtrip works correctly for SIP images.
    """
    i, j = sip_target.pixel_center_meshgrid()
    x, y = sip_target.pixel_to_plane(i, j)
    i2, j2 = sip_target.plane_to_pixel(x, y)

    assert ap.backend.allclose(i, i2, atol=0.05), "i coordinates should match after WCS roundtrip"
    assert ap.backend.allclose(j, j2, atol=0.05), "j coordinates should match after WCS roundtrip"


def test_sip_image_save_load(sip_target):
    """
    Test that SIP images can be saved and loaded correctly.
    """
    # Save the SIP image to a file
    sip_target.save("test_sip_image.fits")

    # Load the SIP image from the file
    loaded_image = ap.SIPTargetImage(filename="test_sip_image.fits")

    # Check that the loaded image matches the original
    assert ap.backend.allclose(
        sip_target.data, loaded_image.data
    ), "Loaded image data should match original"
    assert ap.backend.allclose(
        sip_target.pixelscale, loaded_image.pixelscale
    ), "Loaded image pixelscale should match original"
    assert ap.backend.allclose(
        sip_target.zeropoint, loaded_image.zeropoint
    ), "Loaded image zeropoint should match original"
    print(loaded_image.sipA)
    assert all(
        np.allclose(sip_target.sipA[key], loaded_image.sipA[key]) for key in sip_target.sipA
    ), "Loaded image sipA should match original"
    assert all(
        np.allclose(sip_target.sipB[key], loaded_image.sipB[key]) for key in sip_target.sipB
    ), "Loaded image sipB should match original"
    assert all(
        np.allclose(sip_target.sipAP[key], loaded_image.sipAP[key]) for key in sip_target.sipAP
    ), "Loaded image sipAP should match original"
    assert all(
        np.allclose(sip_target.sipBP[key], loaded_image.sipBP[key]) for key in sip_target.sipBP
    ), "Loaded image sipBP should match original"
