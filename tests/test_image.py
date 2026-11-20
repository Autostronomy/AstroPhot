"""Image, TargetImage, PSFImage and JacobianImage (astrophot/image).

Coordinates are tested against astropy in test_image_wcs.py.
"""

import astrophot as ap
import numpy as np
import pytest

######################################################################
# Image
######################################################################


def test_image_creation(image):
    image.to()
    assert image.pixelscale == 1.0, "image should track pixelscale"
    assert image.zeropoint == 1.0, "image should track zeropoint"
    assert image.crpix[0] == 0, "image should track crpix"
    assert image.crpix[1] == 0, "image should track crpix"

    image.to(dtype=ap.backend.float64)
    slicer = ap.Window((7, 13, 4, 7), image)
    sliced_image = image[slicer]
    assert sliced_image.crpix[0] == -7, "crpix of subimage should give relative position"
    assert sliced_image.crpix[1] == -4, "crpix of subimage should give relative position"
    assert sliced_image._data.shape == (6, 3), "sliced image should have correct shape"


def test_copy(image):
    copy_image = image.copy()
    assert image.pixelscale == copy_image.pixelscale, "copied image should have same pixelscale"
    assert image.zeropoint == copy_image.zeropoint, "copied image should have same zeropoint"
    assert image.window.extent == copy_image.window.extent, "copied image should have same window"
    copy_image += 1
    assert image._data[0][0] == 0.0, "copied image should not share data with original"

    blank_copy_image = image.blank_copy()
    assert (
        image.pixelscale == blank_copy_image.pixelscale
    ), "copied image should have same pixelscale"
    assert image.zeropoint == blank_copy_image.zeropoint, "copied image should have same zeropoint"
    assert (
        image.window.extent == blank_copy_image.window.extent
    ), "copied image should have same window"
    blank_copy_image += 1
    assert image._data[0][0] == 0.0, "copied image should not share data with original"


def test_image_arithmetic(image):
    slicer = ap.Window((-1, 5, 6, 15), image)
    sliced_image = image[slicer]
    sliced_image += 1

    assert image._data[1][8] == 0, "slice should not update base image"
    assert image._data[5][5] == 0, "slice should not update base image"

    second_image = ap.Image(
        data=np.ones((5, 5)),
        pixelscale=1.0,
        zeropoint=1.0,
        crpix=(-1, 1),
    )

    # Test iadd
    image += second_image
    assert image._data[0][0] == 0, "image addition should only update its region"
    assert image._data[3][3] == 1, "image addition should update its region"
    assert image._data[3][4] == 0, "image addition should only update its region"
    assert image._data[5][3] == 1, "image addition should update its region"

    # Test isubtract
    image -= second_image
    assert ap.backend.allclose(
        image.data, ap.backend.zeros_like(image.data)
    ), "image subtraction should only update its region"


def test_image_manipulation():
    new_image = ap.Image(
        data=np.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    # image reduction
    for scale in [2, 4, 8, 16]:
        reduced_image = new_image.reduce(scale)

        assert reduced_image._data[0][0] == scale**2, "reduced image should sum sub pixels"
        assert reduced_image.pixelscale == scale, "pixelscale should increase with reduced image"

    # image cropping
    crop_image = new_image.crop([1])
    assert crop_image._data.shape[1] == 14, "crop should cut 1 pixel from both sides here"
    crop_image = new_image.crop([3, 2])
    assert (
        crop_image._data.shape[0] == 26
    ), "crop should have cut 3 pixels from both sides of this axis"
    crop_image = new_image.crop([3, 2, 1, 0])
    assert (
        crop_image._data.shape[0] == 27
    ), "crop should have cut 3 pixels from left, 2 from right, 1 from top, and 0 from bottom"


def test_image_save_load(tmp_path):
    new_image = ap.Image(
        data=np.ones((16, 32)),
        pixelscale=0.76,
        zeropoint=21.4,
        crtan=(8.0, 1.2),
        crpix=(2, 3),
        crval=(100.0, -32.1),
    )

    new_image.save(tmp_path / "image.fits")

    loaded_image = ap.Image(filename=tmp_path / "image.fits")

    assert ap.backend.allclose(
        new_image.data, loaded_image.data
    ), "Loaded image should have same pixel values"
    assert ap.backend.allclose(
        new_image.crtan.value, loaded_image.crtan.value
    ), "Loaded image should have same tangent plane origin"
    assert np.all(
        new_image.crpix == loaded_image.crpix
    ), "Loaded image should have same reference pixel"
    assert ap.backend.allclose(
        new_image.crval.value, loaded_image.crval.value
    ), "Loaded image should have same reference world coordinates"
    assert ap.backend.allclose(
        new_image.pixelscale, loaded_image.pixelscale
    ), "Loaded image should have same pixel scale"
    assert ap.backend.allclose(
        new_image.CD.value, loaded_image.CD.value
    ), "Loaded image should have same pixel scale"
    assert new_image.zeropoint == loaded_image.zeropoint, "Loaded image should have same zeropoint"


######################################################################
# TargetImage
######################################################################


def test_target_image_variance():
    new_image = ap.TargetImage(
        data=np.ones((16, 32)),
        variance=2 * np.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    assert new_image.variance[0][0] == 2, "target image should store variance"

    reduced_image = new_image.reduce(2)
    assert reduced_image.variance[0][0] == 8, "reduced image should sum sub pixels"

    new_image.variance = None
    assert new_image.variance[0][0] == 1, "target image update to neutral variance"


def test_target_image_mask():
    new_image = ap.TargetImage(
        data=np.ones((16, 32)),
        mask=np.arange(16 * 32).reshape((16, 32)) % 4 == 0,
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert ap.backend.sum(new_image.mask) > 0, "target image should store mask"

    reduced_image = new_image.reduce(2)
    assert reduced_image._mask[0][0] == 1, "reduced image should mask appropriately"
    assert reduced_image._mask[1][0] == 0, "reduced image should mask appropriately"

    new_image.mask = None
    assert ap.backend.sum(new_image.mask) == 0, "target image update to no mask"

    data = np.ones((16, 32))
    data[1, 1] = np.nan
    data[5, 5] = np.nan

    new_image = ap.TargetImage(
        data=data,
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert ap.backend.sum(new_image.mask) > 0, "target image with nans should create mask"
    assert new_image._mask[1][1].item() == True, "nan should be masked"
    assert new_image._mask[5][5].item() == True, "nan should be masked"


def test_target_image_psf():
    new_image = ap.TargetImage(
        data=np.ones((15, 33)),
        psf=np.ones((9, 9)),
        pixelscale=1.0,
        zeropoint=1.0,
    )
    assert new_image.has_psf, "target image should store variance"
    assert new_image.psf.pad == 4, "psf border should be half psf size"

    reduced_image = new_image.reduce(3)
    assert reduced_image.psf.upsample == 3, "reduced image should now have upsampled PSF"

    new_image.psf = None
    assert not new_image.has_psf, "target image update to no psf"


def test_target_image_reduce():
    new_image = ap.TargetImage(
        data=np.ones((30, 36)),
        psf=np.ones((9, 9)),
        variance="auto",
        pixelscale=1.0,
        zeropoint=1.0,
    )
    smaller_image = new_image.reduce(3)
    assert smaller_image._data[0][0] == 9, "reduction should sum flux"
    assert tuple(smaller_image._data.shape) == (12, 10), "reduction should decrease image size"


def test_target_image_save_load(tmp_path):
    new_image = ap.TargetImage(
        data=np.ones((16, 32)),
        variance=np.ones((16, 32)),
        mask=np.zeros((16, 32)),
        psf=np.ones((9, 9)),
        CD=[[1.0, 0.0], [0.0, 1.5]],
        zeropoint=1.0,
    )

    new_image.save(tmp_path / "target.fits")

    loaded_image = ap.TargetImage(filename=tmp_path / "target.fits")

    assert ap.backend.allclose(
        new_image.data, loaded_image.data
    ), "Loaded image should have same pixel values"
    assert ap.backend.allclose(
        new_image.mask, loaded_image.mask
    ), "Loaded image should have same mask"
    assert ap.backend.allclose(
        new_image.variance, loaded_image.variance
    ), "Loaded image should have same variance"
    assert ap.backend.allclose(
        new_image.psf.data, loaded_image.psf.data
    ), "Loaded image should have same psf"
    assert ap.backend.allclose(
        new_image.CD.value, loaded_image.CD.value
    ), "Loaded image should have same pixel scale"


def test_target_image_auto_var(sersic_target):
    sersic_target.variance = "auto"
    variance = ap.backend.to_numpy(sersic_target.variance)
    good = ~ap.backend.to_numpy(sersic_target.mask)
    assert np.all(np.isfinite(variance[good])), "auto variance should be finite"
    assert np.all(variance[good] > 0), "auto variance should be positive"


def test_target_image_errors():
    new_image = ap.TargetImage(
        data=np.ones((16, 32)),
        pixelscale=1.0,
        zeropoint=1.0,
    )

    # bad variance
    with pytest.raises(ap.errors.SpecificationConflict):
        new_image.variance = np.ones((5, 5))

    # bad mask
    with pytest.raises(ap.errors.SpecificationConflict):
        new_image.mask = np.zeros((5, 5))


######################################################################
# PSFImage and JacobianImage
######################################################################


def test_psf_image_copying():
    psf_image = ap.PSFImage(
        data=np.ones((15, 15)),
    )

    assert psf_image.pad == 7, "psf image should have correct psf_pad"
    psf_image.normalize()
    assert np.allclose(
        ap.backend.to_numpy(psf_image._data), 1 / 15**2
    ), "psf image should normalize to sum to 1"


def test_jacobian_add():
    new_image = ap.JacobianImage(
        parameters=["a", "b", "c"],
        data=np.ones((16, 32, 3)),
    )
    other_image = ap.JacobianImage(
        parameters=["b", "d"],
        data=5 * np.ones((4, 4, 2)),
    )

    new_image += other_image

    assert tuple(new_image._data.shape) == (
        32,
        16,
        3,
    ), "Jacobian addition should manage parameter identities"
    assert tuple(new_image.flatten("data").shape) == (
        512,
        3,
    ), "Jacobian should flatten to Npix*Nparams tensor"
    assert new_image._data[0, 0, 0].item() == 1, "Jacobian addition should not change original data"
    assert new_image._data[0, 0, 1].item() == 6, " Jacobian addition should add correctly"
