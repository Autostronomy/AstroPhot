import unittest
import astrophot as ap
import numpy as np
import torch

######################################################################
# Image List Object
######################################################################


class TestImageList(unittest.TestCase):
    def test_image_creation(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
            metadata = {"note":"test image 1"},
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
            metadata = {"note":"test image 2"},
        )

        test_image = ap.image.Image_List((base_image1, base_image2))

        for image, original_image in zip(test_image, (base_image1, base_image2)):
            self.assertEqual(
                image.pixel_length,
                original_image.pixel_length,
                "image should track pixelscale",
            )
            self.assertEqual(
                image.zeropoint,
                original_image.zeropoint,
                "image should track zeropoint",
            )
            self.assertEqual(
                image.origin[0], original_image.origin[0], "image should track origin"
            )
            self.assertEqual(
                image.origin[1], original_image.origin[1], "image should track origin"
            )
            self.assertEqual(image.metadata["note"], original_image.metadata["note"], "image should track note")

        slicer = ap.image.Window_List(
            (ap.image.Window(origin=(3, 2), pixel_shape=(4, 5)), ap.image.Window(origin=(3, 2), pixel_shape=(4, 5)))
        )
        sliced_image = test_image[slicer]

        self.assertEqual(sliced_image[0].origin[0], 3, "image should track origin")
        self.assertEqual(sliced_image[0].origin[1], 2, "image should track origin")
        self.assertEqual(sliced_image[1].origin[0], 3, "image should track origin")
        self.assertEqual(sliced_image[1].origin[1], 2, "image should track origin")
        self.assertEqual(
            base_image1.origin[0], 0, "subimage should not change image origin"
        )
        self.assertEqual(
            base_image1.origin[1], 0, "subimage should not change image origin"
        )

    def test_copy(self):

        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )

        test_image = ap.image.Image_List((base_image1, base_image2))

        copy_image = test_image.copy()
        for ti, ci in zip(test_image, copy_image):
            self.assertEqual(
                ti.pixel_length, ci.pixel_length, "copied image should have same pixelscale"
            )
            self.assertEqual(
                ti.zeropoint, ci.zeropoint, "copied image should have same zeropoint"
            )
            self.assertEqual(
                ti.window, ci.window, "copied image should have same window"
            )
            preval = ti.data[0][0].item()
            ci += 1
            self.assertEqual(
                ti.data[0][0],
                preval,
                "copied image should not share data with original",
            )

        blank_copy_image = test_image.blank_copy()
        for ti, ci in zip(test_image, blank_copy_image):
            self.assertEqual(
                ti.pixel_length, ci.pixel_length, "copied image should have same pixelscale"
            )
            self.assertEqual(
                ti.zeropoint, ci.zeropoint, "copied image should have same zeropoint"
            )
            self.assertEqual(
                ti.window, ci.window, "copied image should have same window"
            )
            preval = ti.data[0][0].item()
            ci += 1
            self.assertEqual(
                ti.data[0][0],
                preval,
                "copied image should not share data with original",
            )

    def test_image_arithmetic(self):

        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )
        test_image = ap.image.Image_List((base_image1, base_image2))

        arr3 = torch.ones((10, 15))
        base_image3 = ap.image.Image(
            data=arr3,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.ones(2),
        )
        arr4 = torch.zeros((15, 10))
        base_image4 = ap.image.Image(
            data=arr4,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.zeros(2),
        )
        second_image = ap.image.Image_List((base_image3, base_image4))

        # Test iadd
        test_image += second_image

        self.assertEqual(
            test_image[0].data[0][0], 0, "image addition should only update its region"
        )
        self.assertEqual(
            test_image[0].data[3][3], 1, "image addition should update its region"
        )
        self.assertEqual(
            test_image[1].data[0][0], 1, "image addition should update its region"
        )
        self.assertEqual(
            test_image[1].data[1][1], 1, "image addition should update its region"
        )

        # Test iadd
        test_image -= second_image

        self.assertEqual(
            test_image[0].data[0][0], 0, "image addition should only update its region"
        )
        self.assertEqual(
            test_image[0].data[3][3], 0, "image addition should update its region"
        )
        self.assertEqual(
            test_image[1].data[0][0], 1, "image addition should update its region"
        )
        self.assertEqual(
            test_image[1].data[1][1], 1, "image addition should update its region"
        )

    def test_image_list_display(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )
        test_image = ap.image.Image_List((base_image1, base_image2))

        self.assertIsInstance(str(test_image), str, "String representation should be a string!")
        self.assertIsInstance(repr(test_image), str, "Repr should be a string!")

    def test_image_list_windowset(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
            note="test image 1",
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
            note="test image 2",
        )
        test_image = ap.image.Image_List((base_image1, base_image2))
        arr3 = torch.ones((10, 15))
        base_image3 = ap.image.Image(
            data=arr3,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.ones(2),
            note="test image 3",
        )
        arr4 = torch.zeros((15, 10))
        base_image4 = ap.image.Image(
            data=arr4,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.zeros(2),
            note="test image 4",
        )
        second_image = ap.image.Image_List((base_image3, base_image4), window = test_image.window)       

    def test_image_list_errors(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )
        test_image = ap.image.Image_List((base_image1, base_image2))
        # Bad ra dec reference point
        bad_base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            reference_radec=torch.ones(2),
        )
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Image_List((base_image1, bad_base_image2))

        # Bad tangent plane x y reference point
        bad_base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            reference_planexy=torch.ones(2),
        )
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Image_List((base_image1, bad_base_image2))

        # Bad WCS projection
        bad_base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            projection="orthographic",
        )
        with self.assertRaises(ap.errors.ConflicingWCS):
            test_image = ap.image.Image_List((base_image1, bad_base_image2))


class TestModelImageList(unittest.TestCase):
    def test_model_image_list_creation(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Model_Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Model_Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )

        test_image = ap.image.Model_Image_List((base_image1, base_image2))

        save_image = test_image.copy()
        second_image = test_image.copy()

        second_image += (2, 2)
        second_image -= (1, 1)

        test_image += second_image

        test_image -= second_image

        self.assertTrue(
            torch.all(test_image[0].data == save_image[0].data),
            "adding then subtracting should give the same image",
        )
        self.assertTrue(
            torch.all(test_image[1].data == save_image[1].data),
            "adding then subtracting should give the same image",
        )

        print(test_image.data)
        test_image.clear_image()
        print(test_image.data)
        test_image.replace(second_image)
        print(test_image.data)

        test_image -= (1, 1)
        print(test_image.data)

        self.assertTrue(
            torch.all(test_image[0].data == save_image[0].data),
            "adding then subtracting should give the same image",
        )
        self.assertTrue(
            torch.all(test_image[1].data == save_image[1].data),
            "adding then subtracting should give the same image",
        )

        self.assertIsNone(
            test_image.target_identity,
            "Targets have not been assigned so target identity should be None",
        )

    def test_errors(self):

        # Model_Image_List with non Model_Image object
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Model_Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Target_Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )

        with self.assertRaises(ap.errors.InvalidImage):
            test_image = ap.image.Model_Image_List((base_image1, base_image2))
        

class TestTargetImageList(unittest.TestCase):
    def test_target_image_list_creation(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Target_Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
            variance=torch.ones_like(arr1),
            mask=torch.zeros_like(arr1),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Target_Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
            variance=torch.ones_like(arr2),
            mask=torch.zeros_like(arr2),
        )

        test_image = ap.image.Target_Image_List((base_image1, base_image2))

        save_image = test_image.copy()
        second_image = test_image.copy()

        second_image += (2, 2)
        second_image -= (1, 1)

        test_image += second_image

        test_image -= second_image

        self.assertTrue(
            torch.all(test_image[0].data == save_image[0].data),
            "adding then subtracting should give the same image",
        )
        self.assertTrue(
            torch.all(test_image[1].data == save_image[1].data),
            "adding then subtracting should give the same image",
        )

        test_image += (1, 1)
        test_image -= (1, 1)

        self.assertTrue(
            torch.all(test_image[0].data == save_image[0].data),
            "adding then subtracting should give the same image",
        )
        self.assertTrue(
            torch.all(test_image[1].data == save_image[1].data),
            "adding then subtracting should give the same image",
        )

    def test_targetlist_errors(self):
        arr1 = torch.zeros((10, 15))
        base_image1 = ap.image.Target_Image(
            data=arr1,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2),
            variance=torch.ones_like(arr1),
            mask=torch.zeros_like(arr1),
        )
        arr2 = torch.ones((15, 10))
        base_image2 = ap.image.Image(
            data=arr2,
            pixelscale=0.5,
            zeropoint=2.0,
            origin=torch.ones(2),
        )
        with self.assertRaises(ap.errors.InvalidImage):
            test_image = ap.image.Target_Image_List((base_image1, base_image2))
        

class TestJacobianImageList(unittest.TestCase):
    def test_jacobian_image_list_creation(self):
        arr1 = torch.zeros((10, 15, 3))
        base_image1 = ap.image.Jacobian_Image(
            data=arr1,
            parameters=["a", "b", "c"],
            target_identity="target1",
            pixelscale=1.0,
            zeropoint=1.0,
            window=ap.image.Window(
                origin=torch.zeros(2) + 0.1, pixel_shape=torch.tensor((15, 10))
            ),
        )
        arr2 = torch.ones((15, 10, 3))
        base_image2 = ap.image.Jacobian_Image(
            data=arr2,
            parameters=["a", "b", "c"],
            target_identity="target2",
            pixelscale=0.5,
            zeropoint=2.0,
            window=ap.image.Window(
                origin=torch.zeros(2) + 0.2, pixel_shape=torch.tensor((10, 15))
            ),
        )

        test_image = ap.image.Jacobian_Image_List((base_image1, base_image2))

        second_image = test_image.copy()

        test_image += second_image

        self.assertEqual(
            test_image.flatten("data").shape,
            (300, 3),
            "flattened jacobian should include all pixels and merge parameters",
        )


if __name__ == "__main__":
    unittest.main()
