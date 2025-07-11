import astrophot as ap
import torch
import numpy as np

from utils import make_basic_sersic
import pytest

######################################################################
# Image Objects
######################################################################


def test_image_creation():
    arr = torch.zeros((10, 15))
    base_image = ap.Image(
        data=arr,
        pixelscale=1.0,
        zeropoint=1.0,
    )

    assert base_image.pixelscale == 1.0, "image should track pixelscale"
    assert base_image.zeropoint == 1.0, "image should track zeropoint"
    assert base_image.crpix[0] == 0, "image should track crpix"
    assert base_image.crpix[1] == 0, "image should track crpix"

    slicer = ap.Window((7, 13, 4, 7), base_image)
    sliced_image = base_image[slicer]
    assert sliced_image.crpix[0] == -7, "crpix of subimage should give relative position"
    assert sliced_image.crpix[1] == -4, "crpix of subimage should give relative position"
    assert sliced_image.shape == (6, 3), "sliced image should have correct shape"


def test_copy():
    new_image = ap.Image(
        data=torch.zeros((10, 15)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    copy_image = new_image.copy()
    assert new_image.pixelscale == copy_image.pixelscale, "copied image should have same pixelscale"
    assert new_image.zeropoint == copy_image.zeropoint, "copied image should have same zeropoint"
    assert (
        new_image.window.extent == copy_image.window.extent
    ), "copied image should have same window"
    copy_image += 1
    assert new_image.data[0][0] == 0.0, "copied image should not share data with original"

    blank_copy_image = new_image.blank_copy()
    assert (
        new_image.pixelscale == blank_copy_image.pixelscale
    ), "copied image should have same pixelscale"
    assert (
        new_image.zeropoint == blank_copy_image.zeropoint
    ), "copied image should have same zeropoint"
    assert (
        new_image.window.extent == blank_copy_image.window.extent
    ), "copied image should have same window"
    blank_copy_image += 1
    assert new_image.data[0][0] == 0.0, "copied image should not share data with original"


def test_image_arithmetic():
    arr = torch.zeros((10, 12))
    base_image = ap.Image(
        data=arr,
        pixelscale=1.0,
        zeropoint=1.0,
    )
    slicer = ap.Window((-1, 5, 6, 15), base_image)
    sliced_image = base_image[slicer]
    sliced_image += 1

    assert base_image.data[1][8] == 0, "slice should not update base image"
    assert base_image.data[5][5] == 0, "slice should not update base image"

    second_image = ap.Image(
        data=torch.ones((5, 5)),
        pixelscale=1.0,
        zeropoint=1.0,
        crpix=(-1, 1),
    )

    # Test iadd
    base_image += second_image
    assert base_image.data[0][0] == 0, "image addition should only update its region"
    assert base_image.data[3][3] == 1, "image addition should update its region"
    assert base_image.data[3][4] == 0, "image addition should only update its region"
    assert base_image.data[5][3] == 1, "image addition should update its region"

    # Test isubtract
    base_image -= second_image
    assert torch.all(
        torch.isclose(base_image.data, torch.zeros_like(base_image.data))
    ), "image subtraction should only update its region"


def test_image_manipulation():
    new_image = ap.Image(
        data=torch.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    # image reduction
    for scale in [2, 4, 8, 16]:
        reduced_image = new_image.reduce(scale)

        assert reduced_image.data[0][0] == scale**2, "reduced image should sum sub pixels"
        assert reduced_image.pixelscale == scale, "pixelscale should increase with reduced image"

    # image cropping
    crop_image = new_image.crop([1])
    assert crop_image.shape[1] == 14, "crop should cut 1 pixel from both sides here"
    crop_image = new_image.crop([3, 2])
    assert (
        crop_image.data.shape[0] == 26
    ), "crop should have cut 3 pixels from both sides of this axis"
    crop_image = new_image.crop([3, 2, 1, 0])
    assert (
        crop_image.data.shape[0] == 27
    ), "crop should have cut 3 pixels from left, 2 from right, 1 from top, and 0 from bottom"


def test_image_save_load():
    new_image = ap.Image(
        data=torch.ones((16, 32)),
        pixelscale=0.76,
        zeropoint=21.4,
        crtan=(8.0, 1.2),
        crpix=(2, 3),
        crval=(100.0, -32.1),
    )

    new_image.save("Test_AstroPhot.fits")

    loaded_image = ap.Image(filename="Test_AstroPhot.fits")

    assert torch.all(
        new_image.data == loaded_image.data
    ), "Loaded image should have same pixel values"
    assert torch.all(
        new_image.crtan.value == loaded_image.crtan.value
    ), "Loaded image should have same tangent plane origin"
    assert np.all(
        new_image.crpix == loaded_image.crpix
    ), "Loaded image should have same reference pixel"
    assert torch.all(
        new_image.crval.value == loaded_image.crval.value
    ), "Loaded image should have same reference world coordinates"
    assert torch.allclose(
        new_image.pixelscale, loaded_image.pixelscale
    ), "Loaded image should have same pixel scale"
    assert torch.allclose(
        new_image.CD.value, loaded_image.CD.value
    ), "Loaded image should have same pixel scale"
    assert new_image.zeropoint == loaded_image.zeropoint, "Loaded image should have same zeropoint"


def test_image_wcs_roundtrip():
    # Minimal input
    I = ap.Image(
        data=torch.zeros((21, 21)),
        zeropoint=22.5,
        crpix=(10, 10),
        crtan=(1.0, -10.0),
        crval=(160.0, 45.0),
        CD=0.05
        * np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]]
        ),
    )

    assert torch.allclose(
        torch.stack(I.world_to_plane(*I.plane_to_world(*I.center))),
        I.center,
    ), "WCS world/plane roundtrip should return input value"
    assert torch.allclose(
        torch.stack(I.pixel_to_plane(*I.plane_to_pixel(*I.center))),
        I.center,
    ), "WCS pixel/plane roundtrip should return input value"
    assert torch.allclose(
        torch.stack(I.world_to_pixel(*I.pixel_to_world(*torch.zeros_like(I.center)))),
        torch.zeros_like(I.center),
        atol=1e-6,
    ), "WCS world/pixel roundtrip should return input value"


def test_target_image_variance():
    new_image = ap.TargetImage(
        data=torch.ones((16, 32)),
        variance=torch.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    assert new_image.has_variance, "target image should store variance"

    reduced_image = new_image.reduce(2)
    assert reduced_image.variance[0][0] == 4, "reduced image should sum sub pixels"

    new_image.variance = None
    assert not new_image.has_variance, "target image update to no variance"


def test_target_image_mask():
    new_image = ap.TargetImage(
        data=torch.ones((16, 32)),
        mask=torch.arange(16 * 32).reshape((16, 32)) % 4 == 0,
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert new_image.has_mask, "target image should store mask"

    reduced_image = new_image.reduce(2)
    assert reduced_image.mask[0][0] == 1, "reduced image should mask appropriately"
    assert reduced_image.mask[1][0] == 0, "reduced image should mask appropriately"

    new_image.mask = None
    assert not new_image.has_mask, "target image update to no mask"

    data = torch.ones((16, 32))
    data[1, 1] = torch.nan
    data[5, 5] = torch.nan

    new_image = ap.TargetImage(
        data=data,
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert new_image.has_mask, "target image with nans should create mask"
    assert new_image.mask[1][1].item() == True, "nan should be masked"
    assert new_image.mask[5][5].item() == True, "nan should be masked"


def test_target_image_psf():
    new_image = ap.TargetImage(
        data=torch.ones((15, 33)),
        psf=torch.ones((9, 9)),
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert new_image.has_psf, "target image should store variance"
    assert new_image.psf.psf_pad == 4, "psf border should be half psf size"

    reduced_image = new_image.reduce(3)
    assert reduced_image.psf.data[0][0] == 9, "reduced image should sum sub pixels in psf"

    new_image.psf = None
    assert not new_image.has_psf, "target image update to no variance"


def test_target_image_reduce():
    new_image = ap.TargetImage(
        data=torch.ones((30, 36)),
        psf=torch.ones((9, 9)),
        variance="auto",
        pixelscale=1.0,
        zeropoint=1.0,
    )
    smaller_image = new_image.reduce(3)
    assert smaller_image.data[0][0] == 9, "reduction should sum flux"
    assert tuple(smaller_image.data.shape) == (12, 10), "reduction should decrease image size"


def test_target_image_save_load():
    new_image = ap.TargetImage(
        data=torch.ones((16, 32)),
        variance=torch.ones((16, 32)),
        mask=torch.zeros((16, 32)),
        psf=torch.ones((9, 9)),
        CD=[[1.0, 0.0], [0.0, 1.5]],
        zeropoint=1.0,
    )

    new_image.save("Test_target_AstroPhot.fits")

    loaded_image = ap.TargetImage(filename="Test_target_AstroPhot.fits")

    assert torch.all(
        new_image.data == loaded_image.data
    ), "Loaded image should have same pixel values"
    assert torch.all(new_image.mask == loaded_image.mask), "Loaded image should have same mask"
    assert torch.all(
        new_image.variance == loaded_image.variance
    ), "Loaded image should have same variance"
    assert torch.all(
        new_image.psf.data == loaded_image.psf.data
    ), "Loaded image should have same psf"
    assert torch.allclose(
        new_image.CD.value, loaded_image.CD.value
    ), "Loaded image should have same pixel scale"


def test_target_image_auto_var():
    target = make_basic_sersic()
    target.variance = "auto"


def test_target_image_errors():
    new_image = ap.TargetImage(
        data=torch.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    # bad variance
    with pytest.raises(ap.errors.SpecificationConflict):
        new_image.variance = np.ones((5, 5))

    # bad mask
    with pytest.raises(ap.errors.SpecificationConflict):
        new_image.mask = np.zeros((5, 5))


def test_psf_image_copying():
    psf_image = ap.PSFImage(
        data=torch.ones((15, 15)),
    )

    assert psf_image.psf_pad == 7, "psf image should have correct psf_pad"
    psf_image.normalize()
    assert np.allclose(
        psf_image.data.detach().cpu().numpy(), 1 / 15**2
    ), "psf image should normalize to sum to 1"


def test_jacobian_add():
    new_image = ap.JacobianImage(
        parameters=["a", "b", "c"],
        data=torch.ones((16, 32, 3)),
    )
    other_image = ap.JacobianImage(
        parameters=["b", "d"],
        data=5 * torch.ones((4, 4, 2)),
    )

    new_image += other_image

    assert tuple(new_image.data.shape) == (
        32,
        16,
        3,
    ), "Jacobian addition should manage parameter identities"
    assert tuple(new_image.flatten("data").shape) == (
        512,
        3,
    ), "Jacobian should flatten to Npix*Nparams tensor"
    assert new_image.data[0, 0, 0].item() == 1, "Jacobian addition should not change original data"
    assert new_image.data[0, 0, 1].item() == 6, " Jacobian addition should add correctly"
