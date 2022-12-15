import unittest
from autoprof import image
import torch

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

    def test_image_arithmetic(self):

        arr = torch.zeros((10,12))
        base_image = image.Model_Image(data = arr, pixelscale = 1.0, zeropoint = 1.0, origin = torch.ones(2), note = 'test image')
        slicer = image.Window((0,0), (5,5))
        sliced_image = base_image[slicer]
        sliced_image += 1
        
        self.assertEqual(base_image.data[1][1], 1, "slice should update base image")
        self.assertEqual(base_image.data[5][5], 0, "slice should only update its region")

        second_image = image.Model_Image(data = torch.ones((5,5)), pixelscale = 1.0, zeropoint = 1.0, origin = [3,3], note = 'second image')
        base_image += second_image
        self.assertEqual(base_image.data[1][1], 1, "image addition should only update its region")
        self.assertEqual(base_image.data[3][3], 2, "image addition should update its region")
        self.assertEqual(base_image.data[5][5], 1, "image addition should update its region")
        self.assertEqual(base_image.data[8][8], 0, "image addition should only update its region")

        base_image.data[6:,6:] += 1.
        
        self.assertEqual(base_image.data[1][1], 1, "array addition should only update its region")
        self.assertEqual(base_image.data[6][6], 2, "array addition should update its region")
        self.assertEqual(base_image.data[8][8], 1, "array addition should update its region")

        
if __name__ == "__main__":
    unittest.main()
        
