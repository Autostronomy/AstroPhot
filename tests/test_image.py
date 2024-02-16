import unittest
from astrophot import image
import astrophot as ap
import torch
import numpy as np

from utils import get_astropy_wcs
######################################################################
# Image Objects
######################################################################


class TestImage(unittest.TestCase):
    def test_image_creation(self):
        arr = torch.zeros((10, 15))
        base_image = image.Image(
            data = arr, pixelscale=1.0, zeropoint=1.0, origin=torch.zeros(2), metadata={"note": "test image"}
        )

        self.assertEqual(base_image.pixel_length, 1.0, "image should track pixelscale")
        self.assertEqual(base_image.zeropoint, 1.0, "image should track zeropoint")
        self.assertEqual(base_image.origin[0], 0, "image should track origin")
        self.assertEqual(base_image.origin[1], 0, "image should track origin")
        self.assertEqual(base_image.metadata["note"], "test image", "image should track note")

        slicer = image.Window(origin = (3, 2), pixel_shape = (4, 5))
        sliced_image = base_image[slicer]
        self.assertEqual(sliced_image.origin[0], 3, "image should track origin")
        self.assertEqual(sliced_image.origin[1], 2, "image should track origin")
        self.assertEqual(
            base_image.origin[0], 0, "subimage should not change image origin"
        )
        self.assertEqual(
            base_image.origin[1], 0, "subimage should not change image origin"
        )

        second_base_image = image.Image(data = arr, pixelscale=1.0, metadata={"note": "test image"})
        self.assertEqual(base_image.pixel_length, 1.0, "image should track pixelscale")
        self.assertIsNone(second_base_image.zeropoint, "image should track zeropoint")
        self.assertEqual(second_base_image.origin[0], 0, "image should track origin")
        self.assertEqual(second_base_image.origin[1], 0, "image should track origin")
        self.assertEqual(
            second_base_image.metadata["note"], "test image", "image should track note"
        )
        
    def test_copy(self):

        new_image = image.Image(
            data = torch.zeros((10, 15)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )

        copy_image = new_image.copy()
        self.assertEqual(
            new_image.pixel_length,
            copy_image.pixel_length,
            "copied image should have same pixelscale",
        )
        self.assertEqual(
            new_image.zeropoint,
            copy_image.zeropoint,
            "copied image should have same zeropoint",
        )
        self.assertEqual(
            new_image.window, copy_image.window, "copied image should have same window"
        )
        copy_image += 1
        self.assertEqual(
            new_image.data[0][0],
            0.0,
            "copied image should not share data with original",
        )

        blank_copy_image = new_image.blank_copy()
        self.assertEqual(
            new_image.pixel_length,
            blank_copy_image.pixel_length,
            "copied image should have same pixelscale",
        )
        self.assertEqual(
            new_image.zeropoint,
            blank_copy_image.zeropoint,
            "copied image should have same zeropoint",
        )
        self.assertEqual(
            new_image.window,
            blank_copy_image.window,
            "copied image should have same window",
        )
        blank_copy_image += 1
        self.assertEqual(
            new_image.data[0][0],
            0.0,
            "copied image should not share data with original",
        )

    def test_image_arithmetic(self):

        arr = torch.zeros((10, 12))
        base_image = image.Image(
            data=arr,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.ones(2),
        )
        slicer = image.Window(origin = (0, 0), pixel_shape = (5, 5))
        sliced_image = base_image[slicer]
        sliced_image += 1

        self.assertEqual(base_image.data[1][1], 1, "slice should update base image")
        self.assertEqual(
            base_image.data[5][5], 0, "slice should only update its region"
        )

        second_image = image.Image(
            data=torch.ones((5, 5)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=[3, 3],
        )

        # Test iadd
        base_image += second_image
        self.assertEqual(
            base_image.data[1][1], 1, "image addition should only update its region"
        )
        self.assertEqual(
            base_image.data[3][3], 2, "image addition should update its region"
        )
        self.assertEqual(
            base_image.data[5][5], 1, "image addition should update its region"
        )
        self.assertEqual(
            base_image.data[8][8], 0, "image addition should only update its region"
        )

        # Test isubtract
        base_image -= second_image
        self.assertEqual(
            base_image.data[1][1], 1, "image subtraction should only update its region"
        )
        self.assertEqual(
            base_image.data[3][3], 1, "image subtraction should update its region"
        )
        self.assertEqual(
            base_image.data[5][5], 0, "image subtraction should update its region"
        )
        self.assertEqual(
            base_image.data[8][8], 0, "image subtraction should only update its region"
        )

        base_image.data[6:, 6:] += 1.0

        self.assertEqual(
            base_image.data[1][1], 1, "array addition should only update its region"
        )
        self.assertEqual(
            base_image.data[6][6], 1, "array addition should update its region"
        )
        self.assertEqual(
            base_image.data[8][8], 1, "array addition should update its region"
        )

    def test_excersize_arithmatic(self):

        arr = torch.zeros((10, 12))
        base_image = image.Image(
            data=arr,
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.ones(2),
        )
        second_image = image.Image(
            data=torch.ones((5, 5)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=[3, 3],
        )

        new_img = base_image + second_image
        new_img = new_img - second_image

        self.assertTrue(torch.allclose(new_img.data, torch.zeros_like(new_img.data)), "addition and subtraction should produce no change")
        
        base_image += second_image
        base_image -= second_image

        self.assertTrue(torch.allclose(base_image.data, torch.zeros_like(base_image.data)), "addition and subtraction should produce no change")

        new_img = base_image + 10.
        new_img = new_img - 10.

        self.assertTrue(torch.allclose(new_img.data, torch.zeros_like(new_img.data)), "addition and subtraction should produce no change")
        
        base_image += 10.
        base_image -= 10.

        self.assertTrue(torch.allclose(base_image.data, torch.zeros_like(base_image.data)), "addition and subtraction should produce no change")
        
    def test_image_manipulation(self):

        new_image = image.Image(
            data = torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )

        # image reduction
        for scale in [2, 4, 8, 16]:
            reduced_image = new_image.reduce(scale)

            self.assertEqual(
                reduced_image.data[0][0],
                scale ** 2,
                "reduced image should sum sub pixels",
            )
            self.assertEqual(
                reduced_image.pixel_length,
                scale,
                "pixelscale should increase with reduced image",
            )
            self.assertEqual(
                reduced_image.origin[0],
                new_image.origin[0],
                "origin should not change with reduced image",
            )
            self.assertEqual(
                reduced_image.shape[0],
                new_image.shape[0],
                "shape should not change with reduced image",
            )

        # iamge cropping
        new_image.crop([torch.tensor(1, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)])
        self.assertEqual(
            new_image.data.shape[0], 14, "crop should cut 1 pixel from both sides here"
        )
        new_image.crop(torch.tensor([3, 2], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device))
        self.assertEqual(
            new_image.data.shape[1],
            24,
            "previous crop and current crop should have cut from this axis",
        )
        new_image.crop(torch.tensor([3, 2, 1, 0], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device))
        self.assertEqual(
            new_image.data.shape[0],
            9,
            "previous crop and current crop should have cut from this axis",
        )

    def test_image_save_load(self):

        new_image = image.Image(
            data =torch.ones((16, 32)),
            pixelscale=0.76,
            zeropoint=21.4,
            origin=torch.zeros(2) + 0.1,
        )

        new_image.save("Test_AstroPhot.fits")

        loaded_image = ap.image.Image(filename="Test_AstroPhot.fits")

        self.assertTrue(
            torch.all(new_image.data == loaded_image.data),
            "Loaded image should have same pixel values",
        )
        self.assertTrue(
            torch.all(new_image.origin == loaded_image.origin),
            "Loaded image should have same origin",
        )
        self.assertEqual(
            new_image.pixel_length,
            loaded_image.pixel_length,
            "Loaded image should have same pixel scale",
        )
        self.assertEqual(
            new_image.zeropoint,
            loaded_image.zeropoint,
            "Loaded image should have same zeropoint",
        )

    def test_image_wcs_roundtrip(self):

        wcs = get_astropy_wcs()
        # Minimial input
        I = ap.image.Image(
            data = torch.zeros((20,20)),
            zeropoint = 22.5,
            wcs = wcs,
        )

        self.assertTrue(torch.allclose(I.world_to_plane(I.plane_to_world(torch.zeros_like(I.window.reference_radec))), torch.zeros_like(I.window.reference_radec)), "WCS world/plane roundtrip should return input value")
        self.assertTrue(torch.allclose(I.pixel_to_plane(I.plane_to_pixel(torch.zeros_like(I.window.reference_radec))), torch.zeros_like(I.window.reference_radec)), "WCS pixel/plane roundtrip should return input value")
        self.assertTrue(torch.allclose(I.world_to_pixel(I.pixel_to_world(torch.zeros_like(I.window.reference_radec))), torch.zeros_like(I.window.reference_radec), atol = 1e-6), "WCS world/pixel roundtrip should return input value")

        self.assertTrue(torch.allclose(I.pixel_to_plane_delta(I.plane_to_pixel_delta(torch.ones_like(I.window.reference_radec))), torch.ones_like(I.window.reference_radec)), "WCS pixel/plane delta roundtrip should return input value")
        
    def test_image_display(self):
        new_image = image.Image(
            data =torch.ones((16, 32)),
            pixelscale=0.76,
            zeropoint=21.4,
            origin=torch.zeros(2) + 0.1,
        )

        self.assertIsInstance(str(new_image), str, "String representation should be a string!")
        self.assertIsInstance(repr(new_image), str, "Repr should be a string!")

    def test_image_errors(self):

        new_image = image.Image(
            data =torch.ones((16, 32)),
            pixelscale=0.76,
            zeropoint=21.4,
            origin=torch.zeros(2) + 0.1,
        )

        # Change data badly
        with self.assertRaises(ap.errors.SpecificationConflict):
            new_image.data = np.zeros((5,5))

        # Fractional image reduction
        with self.assertRaises(ap.errors.SpecificationConflict):
            reduced = new_image.reduce(0.2)

        # Negative expand image
        with self.assertRaises(ap.errors.SpecificationConflict):
            unexpanded = new_image.expand((-2, 3))
            
class TestTargetImage(unittest.TestCase):
    def test_variance(self):

        new_image = image.Target_Image(
            data=torch.ones((16, 32)),
            variance=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )

        self.assertTrue(new_image.has_variance, "target image should store variance")

        reduced_image = new_image.reduce(2)
        self.assertEqual(
            reduced_image.variance[0][0], 4, "reduced image should sum sub pixels"
        )

        new_image.to()
        new_image.variance = None
        self.assertFalse(new_image.has_variance, "target image update to no variance")

    def test_mask(self):

        new_image = image.Target_Image(
            data=torch.ones((16, 32)),
            mask=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        self.assertTrue(new_image.has_mask, "target image should store mask")

        reduced_image = new_image.reduce(2)
        self.assertEqual(
            reduced_image.mask[0][0], 1, "reduced image should mask apropriately"
        )

        new_image.mask = None
        self.assertFalse(new_image.has_mask, "target image update to no mask")

    def test_psf(self):

        new_image = image.Target_Image(
            data=torch.ones((15, 33)),
            psf=torch.ones((9, 9)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        self.assertTrue(new_image.has_psf, "target image should store variance")
        self.assertEqual(
            new_image.psf.psf_border_int[0],
            5,
            "psf border should be half psf size, rounded up ",
        )

        reduced_image = new_image.reduce(3)
        self.assertEqual(
            reduced_image.psf.data[0][0], 9, "reduced image should sum sub pixels in psf"
        )

        new_image.psf = None
        self.assertFalse(new_image.has_psf, "target image update to no variance")

    def test_reduce(self):
        new_image = image.Target_Image(
            data=torch.ones((30, 36)),
            psf=torch.ones((9, 9)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        smaller_image = new_image.reduce(3)
        self.assertEqual(smaller_image.data[0][0], 9, "reduction should sum flux")
        self.assertEqual(tuple(smaller_image.data.shape), (10,12), "reduction should decrease image size")
        self.assertEqual(smaller_image.psf.data[0][0], 9, "reduction should sum psf flux")
        self.assertEqual(tuple(smaller_image.psf.data.shape), (3,3), "reduction should decrease psf image size")
        
    def test_target_save_load(self):
        new_image = image.Target_Image(
            data=torch.ones((16, 32)),
            variance=torch.ones((16, 32)),
            psf=torch.ones((9, 9)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )

        new_image.save("Test_target_AstroPhot.fits")

        loaded_image = ap.image.Target_Image(filename="Test_target_AstroPhot.fits")

        self.assertTrue(
            torch.all(new_image.variance == loaded_image.variance),
            "Loaded image should have same variance",
        )
        self.assertTrue(
            torch.all(new_image.psf.data == loaded_image.psf.data),
            "Loaded image should have same psf",
        )
    def test_target_errors(self):
        new_image = image.Target_Image(
            data=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )

        # bad variance
        with self.assertRaises(ap.errors.SpecificationConflict):
            new_image.variance = np.ones((5,5))
            
        # bad mask
        with self.assertRaises(ap.errors.SpecificationConflict):
            new_image.mask = np.zeros((5,5))

class TestPSFImage(unittest.TestCase):
    def test_copying(self):
        psf_image = image.PSF_Image(
            data = torch.ones((15,15)),
            pixelscale = 1.,
        )

        copy_psf = psf_image.copy()
        self.assertEqual(psf_image.data[0][0], copy_psf.data[0][0], "copied image should have same data")
        blank_psf = psf_image.blank_copy()
        self.assertNotEqual(psf_image.data[0][0], blank_psf.data[0][0], "blank copied image should not have same data")

        psf_image.to(dtype = torch.float32)

    def test_reducing(self):
        psf_image = image.PSF_Image(
            data = torch.ones((15,15)),
            pixelscale = 1.,
        )
        new_image = image.Target_Image(
            data=torch.ones((36, 45)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
            psf = psf_image,
        )

        reduce_image = new_image.reduce(3)
        self.assertEqual(tuple(reduce_image.psf.data.shape), (5,5), "reducing image should reduce psf")
        self.assertEqual(reduce_image.psf.pixel_length, 3, "reducing image should update pixelscale factor")

    def test_psf_errors(self):
        with self.assertRaises(ap.errors.SpecificationConflict):
            psf_image = image.PSF_Image(
                data = torch.ones((18,15)),
                pixelscale = 1.,
            )
            
        
class TestModelImage(unittest.TestCase):
    def test_replace(self):
        new_image = image.Model_Image(
            data=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        other_image = image.Model_Image(
            data=5 * torch.ones((4, 4)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 4 + 0.1,
        )

        new_image.replace(other_image)
        new_image.replace(other_image.window, other_image.data)

        self.assertEqual(
            new_image.data[0][0],
            1,
            "image replace should occur at proper location in image, this data should be untouched",
        )
        self.assertEqual(
            new_image.data[5][5], 5, "image replace should update values in its window"
        )

        

    def test_shift(self):

        new_image = image.Model_Image(
            data=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        new_image.shift_origin(torch.tensor((-0.1, -0.1), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), is_prepadded=False)

        self.assertAlmostEqual(
            torch.sum(new_image.data).item(),
            16 * 32,
            delta=1,
            msg="Shifting field of ones should give field of ones",
        )

    def test_errors(self):

        with self.assertRaises(ap.errors.InvalidData):
            new_image = image.Model_Image()

class TestJacobianImage(unittest.TestCase):
    def test_jacobian_add(self):

        new_image = ap.image.Jacobian_Image(
            parameters=["a", "b", "c"],
            target_identity="target1",
            data=torch.ones((16, 32, 3)),
            pixelscale=1.0,
            zeropoint=1.0,
            window=ap.image.Window(
                origin=torch.zeros(2) + 0.1, pixel_shape=torch.tensor((32, 16))
            ),
        )
        other_image = ap.image.Jacobian_Image(
            parameters=["b", "d"],
            target_identity="target1",
            data=5 * torch.ones((4, 4, 2)),
            pixelscale=1.0,
            zeropoint=1.0,
            window=ap.image.Window(
                origin=torch.zeros(2) + 4 + 0.1, pixel_shape=torch.tensor((4, 4))
            ),
        )

        new_image += other_image

        self.assertEqual(
            tuple(new_image.data.shape),
            (16, 32, 4),
            "Jacobian addition should manage parameter identities",
        )
        self.assertEqual(
            tuple(new_image.flatten("data").shape),
            (512, 4),
            "Jacobian should flatten to Npix*Nparams tensor",
        )

    def test_jacobian_error(self):

        # Create parameter list with multiple same entries
        with self.assertRaises(ap.errors.SpecificationConflict):
            new_image = ap.image.Jacobian_Image(
                parameters=["a", "b", "c", "a"],
                target_identity="target1",
                data=torch.ones((16, 32, 3)),
                pixelscale=1.0,
                zeropoint=1.0,
                window=ap.image.Window(
                    origin=torch.zeros(2) + 0.1, pixel_shape=torch.tensor((32, 16))
                ),
            )

        # Adding a model image to a jacobian image
        new_image = ap.image.Jacobian_Image(
            parameters=["a", "b", "c"],
            target_identity="target1",
            data=torch.ones((16, 32, 3)),
            pixelscale=1.0,
            zeropoint=1.0,
            window=ap.image.Window(
                origin=torch.zeros(2) + 0.1, pixel_shape=torch.tensor((32, 16))
            ),
        )
        bad_image = image.Model_Image(
            data=torch.ones((16, 32)),
            pixelscale=1.0,
            zeropoint=1.0,
            origin=torch.zeros(2) + 0.1,
        )
        with self.assertRaises(ap.errors.InvalidImage):
            new_image += bad_image
            


if __name__ == "__main__":
    unittest.main()
