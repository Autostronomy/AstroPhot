import astrophot as ap
import numpy as np
import torch
import pytest

######################################################################
# Image List Object
######################################################################


def test_image_creation():
    arr1 = torch.zeros((10, 15))
    base_image1 = ap.Image(data=arr1, pixelscale=1.0, zeropoint=1.0, name="image1")
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(data=arr2, pixelscale=0.5, zeropoint=2.0, name="image2")

    test_image = ap.ImageList((base_image1, base_image2))

    slicer = ap.WindowList(
        (ap.Window((3, 12, 5, 8), base_image1), ap.Window((4, 8, 3, 13), base_image2))
    )
    sliced_image = test_image[slicer]
    print(sliced_image[0].shape, sliced_image[1].shape)
    assert sliced_image[0].shape == (9, 3), "image slice incorrect shape"
    assert sliced_image[1].shape == (4, 10), "image slice incorrect shape"
    assert np.all(sliced_image[0].crpix == np.array([-3, -5])), "image should track origin"
    assert np.all(sliced_image[1].crpix == np.array([-4, -3])), "image should track origin"


def test_copy():
    arr1 = torch.zeros((10, 15)) + 2
    base_image1 = ap.Image(data=arr1, pixelscale=1.0, zeropoint=1.0, name="image1")
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(data=arr2, pixelscale=0.5, zeropoint=2.0, name="image2")

    test_image = ap.ImageList((base_image1, base_image2))

    copy_image = test_image.copy()
    copy_image.images[0] += 5
    copy_image.images[1] += 5

    for ti, ci in zip(test_image, copy_image):
        assert ti.pixelscale == ci.pixelscale, "copied image should have same pixelscale"
        assert ti.zeropoint == ci.zeropoint, "copied image should have same zeropoint"
        assert torch.all(ti.data != ci.data), "copied image should not modify original data"

    blank_copy_image = test_image.blank_copy()
    for ti, ci in zip(test_image, blank_copy_image):
        assert ti.pixelscale == ci.pixelscale, "copied image should have same pixelscale"
        assert ti.zeropoint == ci.zeropoint, "copied image should have same zeropoint"


def test_image_arithmetic():
    arr1 = torch.zeros((10, 15))
    base_image1 = ap.Image(data=arr1, pixelscale=1.0, zeropoint=1.0, name="image1")
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(data=arr2, pixelscale=0.5, zeropoint=2.0, name="image2")
    test_image = ap.ImageList((base_image1, base_image2))

    base_image3 = base_image1.copy()
    base_image3 += 1
    base_image4 = base_image2.copy()
    base_image4 -= 2
    second_image = ap.ImageList((base_image3, base_image4))

    # Test iadd
    test_image += second_image

    assert torch.allclose(
        test_image[0].data, torch.ones_like(base_image1.data)
    ), "image addition should update its region"
    assert torch.allclose(
        base_image1.data, torch.ones_like(base_image1.data)
    ), "image addition should update its region"
    assert torch.allclose(
        test_image[1].data, torch.zeros_like(base_image2.data)
    ), "image addition should update its region"
    assert torch.allclose(
        base_image2.data, torch.zeros_like(base_image2.data)
    ), "image addition should update its region"

    # Test isub
    test_image -= second_image

    assert torch.allclose(
        test_image[0].data, torch.zeros_like(base_image1.data)
    ), "image addition should update its region"
    assert torch.allclose(
        base_image1.data, torch.zeros_like(base_image1.data)
    ), "image addition should update its region"
    assert torch.allclose(
        test_image[1].data, torch.ones_like(base_image2.data)
    ), "image addition should update its region"
    assert torch.allclose(
        base_image2.data, torch.ones_like(base_image2.data)
    ), "image addition should update its region"


def test_model_image_list_error():
    arr1 = torch.zeros((10, 15))
    base_image1 = ap.ModelImage(data=arr1, pixelscale=1.0, zeropoint=1.0)
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(data=arr2, pixelscale=0.5, zeropoint=2.0)

    with pytest.raises(ap.errors.InvalidImage):
        ap.ModelImageList((base_image1, base_image2))


def test_target_image_list_creation():
    arr1 = torch.zeros((10, 15))
    base_image1 = ap.TargetImage(
        data=arr1,
        pixelscale=1.0,
        zeropoint=1.0,
        variance=torch.ones_like(arr1),
        mask=torch.zeros_like(arr1),
        name="image1",
    )
    arr2 = torch.ones((15, 10))
    base_image2 = ap.TargetImage(
        data=arr2,
        pixelscale=0.5,
        zeropoint=2.0,
        variance=torch.ones_like(arr2),
        mask=torch.zeros_like(arr2),
        name="image2",
    )

    test_image = ap.TargetImageList((base_image1, base_image2))

    save_image = test_image.copy()
    second_image = test_image.copy()

    second_image[0].data += 1
    second_image[1].data += 1

    test_image += second_image
    test_image -= second_image

    assert torch.all(
        test_image[0].data == save_image[0].data
    ), "adding then subtracting should give the same image"
    assert torch.all(
        test_image[1].data == save_image[1].data
    ), "adding then subtracting should give the same image"


def test_targetlist_errors():
    arr1 = torch.zeros((10, 15))
    base_image1 = ap.TargetImage(
        data=arr1,
        pixelscale=1.0,
        zeropoint=1.0,
        variance=torch.ones_like(arr1),
        mask=torch.zeros_like(arr1),
    )
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(
        data=arr2,
        pixelscale=0.5,
        zeropoint=2.0,
    )
    with pytest.raises(ap.errors.InvalidImage):
        ap.TargetImageList((base_image1, base_image2))


def test_jacobian_image_list_error():
    arr1 = torch.zeros((10, 15, 3))
    base_image1 = ap.JacobianImage(
        parameters=["a", "1", "zz"], data=arr1, pixelscale=1.0, zeropoint=1.0
    )
    arr2 = torch.ones((15, 10))
    base_image2 = ap.Image(data=arr2, pixelscale=0.5, zeropoint=2.0)

    with pytest.raises(ap.errors.InvalidImage):
        ap.JacobianImageList((base_image1, base_image2))
