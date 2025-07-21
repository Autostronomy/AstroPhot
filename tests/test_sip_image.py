import astrophot as ap
import torch
import numpy as np

from utils import make_basic_sersic
import pytest

######################################################################
# Image Objects
######################################################################


@pytest.fixture()
def sip_target():
    arr = torch.zeros((10, 15))
    return ap.SIPTargetImage(
        data=arr,
        pixelscale=1.0,
        zeropoint=1.0,
        sipA={(1, 0): 1e-4, (0, 1): 1e-4, (2, 3): -1e-5},
        sipB={(1, 0): -1e-4, (0, 1): 5e-5, (2, 3): 2e-6},
        sipAP={(1, 0): -1e-4, (0, 1): -1e-4, (2, 3): 1e-5},
        sipBP={(1, 0): 1e-4, (0, 1): -5e-5, (2, 3): -2e-6},
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


def test_sip_image_wcs_roundtrip(sip_target):
    """
    Test that the WCS roundtrip works correctly for SIP images.
    """
    i, j = sip_target.pixel_center_meshgrid()
    x, y = sip_target.pixel_to_plane(i, j)
    i2, j2 = sip_target.plane_to_pixel(x, y)

    assert torch.allclose(i, i2, atol=0.5), "i coordinates should match after WCS roundtrip"
    assert torch.allclose(j, j2, atol=0.5), "j coordinates should match after WCS roundtrip"
