"""ImageList, TargetImageList and the other lists of images (astrophot/image).

The `image_pair` fixture gives two images of different shape: `image1` is all
zeros and `image2` is all ones.
"""

import astrophot as ap
import numpy as np
import pytest


def test_image_list_slicing(image_pair):
    image1, image2 = image_pair
    test_image = ap.ImageList(image_pair)

    slicer = ap.WindowList((ap.Window((3, 12, 5, 8), image1), ap.Window((4, 8, 3, 13), image2)))
    sliced_image = test_image[slicer]
    assert sliced_image[0]._data.shape == (9, 3), "image slice incorrect shape"
    assert sliced_image[1]._data.shape == (4, 10), "image slice incorrect shape"
    assert np.all(sliced_image[0].crpix == np.array([-3, -5])), "image should track origin"
    assert np.all(sliced_image[1].crpix == np.array([-4, -3])), "image should track origin"


def test_image_list_copy(image_pair):
    test_image = ap.ImageList(image_pair)

    copy_image = test_image.copy()
    copy_image.images[0] += 5
    copy_image.images[1] += 5

    for ti, ci in zip(test_image, copy_image):
        assert ti.pixelscale == ci.pixelscale, "copied image should have same pixelscale"
        assert ti.zeropoint == ci.zeropoint, "copied image should have same zeropoint"
        assert ap.backend.all(ti.data != ci.data), "copied image should not modify original data"

    blank_copy_image = test_image.blank_copy()
    for ti, ci in zip(test_image, blank_copy_image):
        assert ti.pixelscale == ci.pixelscale, "copied image should have same pixelscale"
        assert ti.zeropoint == ci.zeropoint, "copied image should have same zeropoint"


def test_image_list_arithmetic(image_pair):
    image1, image2 = image_pair
    test_image = ap.ImageList(image_pair)

    image3 = image1.copy()
    image3 += 1
    image4 = image2.copy()
    image4 -= 2
    second_image = ap.ImageList((image3, image4))  # ones and minus ones

    def assert_values(value1, value2, message):
        # the list holds the images themselves, so both views must change
        for image in (test_image[0], image1):
            assert ap.backend.allclose(
                image.data, value1 * ap.backend.ones_like(image.data)
            ), message
        for image in (test_image[1], image2):
            assert ap.backend.allclose(
                image.data, value2 * ap.backend.ones_like(image.data)
            ), message

    test_image += second_image
    assert_values(1, 0, "image addition should update its region")

    test_image -= second_image
    assert_values(0, 1, "image subtraction should update its region")

    new_image = test_image + second_image
    new_image = test_image - second_image
    new_image = new_image.to(dtype=ap.backend.float32, device="cpu")
    assert isinstance(new_image, ap.ImageList), "new image should be an ImageList"

    # a single image is matched to its entry in the list
    new_image += image1
    new_image -= image2


def test_target_image_list_arithmetic(target_image_pair):
    test_image = ap.TargetImageList(target_image_pair)

    save_image = test_image.copy()
    second_image = test_image.copy()

    second_image[0].data += 1
    second_image[1].data += 1

    test_image += second_image
    test_image -= second_image

    for ti, si in zip(test_image, save_image):
        assert ap.backend.all(
            ti.data == si.data
        ), "adding then subtracting should give the same image"


@pytest.mark.parametrize(
    "list_type, first",
    [
        (ap.ModelImageList, lambda: ap.ModelImage(data=np.zeros((10, 15)), pixelscale=1.0)),
        (ap.TargetImageList, lambda: ap.TargetImage(data=np.zeros((10, 15)), pixelscale=1.0)),
        (
            ap.JacobianImageList,
            lambda: ap.JacobianImage(
                parameters=["a", "1", "zz"], data=np.zeros((10, 15, 3)), pixelscale=1.0
            ),
        ),
    ],
)
def test_image_list_wrong_type(list_type, first, image_pair):
    """Each kind of list only accepts its own kind of image."""
    with pytest.raises(ap.errors.InvalidImage):
        list_type((first(), image_pair[1]))
