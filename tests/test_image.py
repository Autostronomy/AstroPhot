import unittest
from autoprof import image
import autoprof as ap
import torch

######################################################################
# Image Objects
######################################################################

class TestImage(unittest.TestCase):
    def test_image_creation(self):
        arr = torch.zeros((10,15))
        base_image = image.BaseImage(arr, pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2), note = 'test image')

        self.assertEqual(base_image.pixelscale, 1.0, 'image should track pixelscale')
        self.assertEqual(base_image.zeropoint, 1.0, 'image should track zeropoint')
        self.assertEqual(base_image.origin[0], 0, 'image should track origin')
        self.assertEqual(base_image.origin[1], 0, 'image should track origin')
        self.assertEqual(base_image.note, 'test image', 'image should track note')

        slicer = image.Window((3,2), (4,5))
        sliced_image = base_image[slicer]
        self.assertEqual(sliced_image.origin[0], 3, 'image should track origin')
        self.assertEqual(sliced_image.origin[1], 2, 'image should track origin')
        self.assertEqual(base_image.origin[0], 0, 'subimage should not change image origin')
        self.assertEqual(base_image.origin[1], 0, 'subimage should not change image origin')

        second_base_image = image.BaseImage(arr, pixelscale = 1.0, note = 'test image')
        self.assertEqual(base_image.pixelscale, 1.0, 'image should track pixelscale')
        self.assertIsNone(second_base_image.zeropoint, 'image should track zeropoint')
        self.assertEqual(second_base_image.origin[0], 0, 'image should track origin')
        self.assertEqual(second_base_image.origin[1], 0, 'image should track origin')
        self.assertEqual(second_base_image.note, 'test image', 'image should track note')

    def test_alignment(self):

        new_image = image.BaseImage(torch.zeros((10,15)), pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2) + 0.1, note = 'test image')
        self.assertTrue(new_image.center_alignment()[0].item(), "pixel alignment has wrong sign")
        self.assertFalse(new_image.center_alignment()[1].item(), "pixel alignment has wrong sign")

        self.assertAlmostEqual(new_image.pixel_center_alignment()[0].item(), 0.6, 5, "pixel center alignment mismatch")

    def test_copy(self):

        new_image = image.BaseImage(torch.zeros((10,15)), pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2) + 0.1, note = 'test image')

        copy_image = new_image.copy()
        self.assertEqual(new_image.pixelscale, copy_image.pixelscale, "copied image should have same pixelscale")
        self.assertEqual(new_image.zeropoint, copy_image.zeropoint, "copied image should have same zeropoint")
        self.assertEqual(new_image.window, copy_image.window, "copied image should have same window")
        copy_image += 1
        self.assertEqual(new_image.data[0][0], 0., "copied image should not share data with original")

        blank_copy_image = new_image.blank_copy()
        self.assertEqual(new_image.pixelscale, blank_copy_image.pixelscale, "copied image should have same pixelscale")
        self.assertEqual(new_image.zeropoint, blank_copy_image.zeropoint, "copied image should have same zeropoint")
        self.assertEqual(new_image.window, blank_copy_image.window, "copied image should have same window")
        blank_copy_image += 1
        self.assertEqual(new_image.data[0][0], 0., "copied image should not share data with original")
        
    def test_image_arithmetic(self):

        arr = torch.zeros((10,12))
        base_image = image.BaseImage(data = arr, pixelscale = 1.0, zeropoint = 1.0, origin = torch.ones(2), note = 'test image')
        slicer = image.Window((0,0), (5,5))
        sliced_image = base_image[slicer]
        sliced_image += 1
        
        self.assertEqual(base_image.data[1][1], 1, "slice should update base image")
        self.assertEqual(base_image.data[5][5], 0, "slice should only update its region")

        second_image = image.BaseImage(data = torch.ones((5,5)), pixelscale = 1.0, zeropoint = 1.0, origin = [3,3], note = 'second image')

        # Test iadd
        base_image += second_image
        self.assertEqual(base_image.data[1][1], 1, "image addition should only update its region")
        self.assertEqual(base_image.data[3][3], 2, "image addition should update its region")
        self.assertEqual(base_image.data[5][5], 1, "image addition should update its region")
        self.assertEqual(base_image.data[8][8], 0, "image addition should only update its region")

        # Test isubtract
        base_image -= second_image
        self.assertEqual(base_image.data[1][1], 1, "image subtraction should only update its region")
        self.assertEqual(base_image.data[3][3], 1, "image subtraction should update its region")
        self.assertEqual(base_image.data[5][5], 0, "image subtraction should update its region")
        self.assertEqual(base_image.data[8][8], 0, "image subtraction should only update its region")

        base_image.data[6:,6:] += 1.
        
        self.assertEqual(base_image.data[1][1], 1, "array addition should only update its region")
        self.assertEqual(base_image.data[6][6], 1, "array addition should update its region")
        self.assertEqual(base_image.data[8][8], 1, "array addition should update its region")

    def test_image_manipulation(self):

        new_image = image.BaseImage(torch.ones((16,32)), pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2) + 0.1, note = 'test image')

        # image reduction
        for scale in [2,4,8,16]:
            reduced_image = new_image.reduce(scale)

            self.assertEqual(reduced_image.data[0][0], scale**2, "reduced image should sum sub pixels")
            self.assertEqual(reduced_image.pixelscale, scale, "pixelscale should increase with reduced image")
            self.assertEqual(reduced_image.origin[0], new_image.origin[0], "origin should not change with reduced image")
            self.assertEqual(reduced_image.shape[0], new_image.shape[0], "shape should not change with reduced image")

        # iamge cropping
        new_image.crop([1])
        self.assertEqual(new_image.data.shape[0], 14, "crop should cut 1 pixel from both sides here")
        new_image.crop([3, 2])
        self.assertEqual(new_image.data.shape[1], 24, "previous crop and current crop should have cut from this axis")
        new_image.crop([3, 2, 1, 0])
        self.assertEqual(new_image.data.shape[0], 9, "previous crop and current crop should have cut from this axis")

    def test_image_save_load(self):
        
        new_image = image.BaseImage(torch.ones((16,32)), pixelscale = 0.76, zeropoint = 21.4, origin = torch.zeros(2) + 0.1, note = 'test image')

        new_image.save("Test_AutoProf.fits")

        loaded_image = ap.image.BaseImage(filename = "Test_AutoProf.fits")

        self.assertTrue(torch.all(new_image.data == loaded_image.data), "Loaded image should have same pixel values")
        self.assertTrue(torch.all(new_image.origin == loaded_image.origin), "Loaded image should have same origin")
        self.assertEqual(new_image.pixelscale, loaded_image.pixelscale, "Loaded image should have same pixel scale")
        self.assertEqual(new_image.zeropoint, loaded_image.zeropoint, "Loaded image should have same zeropoint")
        

class TestTargetImage(unittest.TestCase):
    def test_variance(self):
        
        new_image = image.Target_Image(
            data = torch.ones((16,32)),
            variance = torch.ones((16,32)),
            pixelscale = 1.0,
            zeropoint = 1.0,
            origin = torch.zeros(2) + 0.1,
            note = 'test image'
        )
        
        self.assertTrue(new_image.has_variance, "target image should store variance")

        reduced_image = new_image.reduce(2)
        self.assertEqual(reduced_image.variance[0][0], 4, "reduced image should sum sub pixels")
        
        new_image.variance = None
        self.assertFalse(new_image.has_variance, "target image update to no variance")

    def test_mask(self):
        
        new_image = image.Target_Image(
            data = torch.ones((16,32)),
            mask = torch.ones((16,32)),
            pixelscale = 1.0,
            zeropoint = 1.0,
            origin = torch.zeros(2) + 0.1,
            note = 'test image'
        )
        self.assertTrue(new_image.has_mask, "target image should store mask")

        reduced_image = new_image.reduce(2)
        self.assertEqual(reduced_image.mask[0][0], 1, "reduced image should mask apropriately")
        
        new_image.mask = None
        self.assertFalse(new_image.has_mask, "target image update to no mask")

    def test_psf(self):
        
        new_image = image.Target_Image(
            data = torch.ones((16,32)),
            psf = torch.ones((9,9)),
            pixelscale = 1.0,
            zeropoint = 1.0,
            origin = torch.zeros(2) + 0.1,
            note = 'test image'
        )
        self.assertTrue(new_image.has_psf, "target image should store variance")
        self.assertEqual(new_image.psf_border_int[0], 5, "psf border should be half psf size, rounded up ")
        self.assertEqual(new_image.psf_border[0], 5, "psf border should be half psf size, rounded up ")

        reduced_image = new_image.reduce(3)
        self.assertEqual(reduced_image.psf[0][0], 9, "reduced image should sum sub pixels in psf")
        
        new_image.psf = None
        self.assertFalse(new_image.has_psf, "target image update to no variance")

    def test_target_save_load(self):
        new_image = image.Target_Image(
            data = torch.ones((16,32)),
            variance = torch.ones((16,32)),
            psf = torch.ones((9,9)),
            pixelscale = 1.0,
            zeropoint = 1.0,
            origin = torch.zeros(2) + 0.1,
            note = 'test image'
        )
        
        new_image.save("Test_target_AutoProf.fits")

        loaded_image = ap.image.Target_Image(filename = "Test_target_AutoProf.fits")

        self.assertTrue(torch.all(new_image.variance == loaded_image.variance), "Loaded image should have same variance")
        self.assertTrue(torch.all(new_image.psf == loaded_image.psf), "Loaded image should have same psf")
        
class TestModelImage(unittest.TestCase):
    def test_replace(self):
        new_image = image.Model_Image(data = torch.ones((16,32)), pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2) + 0.1, note = 'test image')
        other_image = image.Model_Image(data = 5*torch.ones((4,4)), pixelscale = 1.0, zeropoint = 1.0, origin = torch.zeros(2) + 4 + 0.1, note = 'other image')

        new_image.replace(other_image)

        self.assertEqual(new_image.data[0][0], 1, "image replace should occur at proper location in image, this data should be untouched")
        self.assertEqual(new_image.data[5][5], 5, "image replace should update values in its window")
        
        
if __name__ == "__main__":
    unittest.main()
        
