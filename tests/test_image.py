import unittest
from autoprof import image
import numpy as np

class TestImage(unittest.TestCase):
    def test_image_creation(self):
        arr = np.zeros((10,10))
        base_image = image.AP_Image(arr, pixelscale = 1.0, zeropoint = 1.0, rotation = 0.0, origin = np.zeros(2, dtype = int), note = 'test image')

        self.assertEqual(base_image.pixelscale, 1.0, 'image should track pixelscale')
        self.assertEqual(base_image.zeropoint, 1.0, 'image should track zeropoint')
        self.assertEqual(base_image.rotation, 0.0, 'image should track rotation')
        self.assertEqual(base_image.origin[0], 0, 'image should track origin')
        self.assertEqual(base_image.origin[1], 0, 'image should track origin')
        self.assertEqual(base_image.note, 'test image', 'image should track note')
        
        sliced_image = base_image.subimage(3, 7, 2, 7)
        self.assertEqual(sliced_image.origin[0], 3, 'image should track origin')
        self.assertEqual(sliced_image.origin[1], 2, 'image should track origin')
        self.assertEqual(base_image.origin[0], 0, 'subimage should not change image origin')
        self.assertEqual(base_image.origin[1], 0, 'subimage should not change image origin')

        second_base_image = image.AP_Image(arr, pixelscale = 1.0, note = 'test image')
        self.assertEqual(base_image.pixelscale, 1.0, 'image should track pixelscale')
        self.assertIsNone(second_base_image.zeropoint, 'image should track zeropoint')
        self.assertIsNone(second_base_image.rotation, 'image should track rotation')
        self.assertEqual(second_base_image.origin[0], 0, 'image should track origin')
        self.assertEqual(second_base_image.origin[1], 0, 'image should track origin')
        self.assertEqual(second_base_image.note, 'test image', 'image should track note')

    def test_image_arithmetic(self):

        arr = np.zeros((10,10))
        base_image = image.AP_Image(arr, pixelscale = 1.0, zeropoint = 1.0, rotation = 0.0, origin = np.ones(2, dtype = int), note = 'test image')
        sliced_image = base_image.subimage(0,5,0,5)
        sliced_image += 1
        
        self.assertEqual(base_image[1][1], 1, "slice should update base image")
        self.assertEqual(base_image[5][5], 0, "slice should only update its region")

        second_image = image.AP_Image(np.ones((5,5)), pixelscale = 1.0, zeropoint = 1.0, rotation = 0.0, origin = [3,3], note = 'second image')
        base_image.add_image(second_image)
        
        self.assertEqual(base_image[1][1], 1, "image addition should only update its region")
        self.assertEqual(base_image[3][3], 2, "image addition should update its region")
        self.assertEqual(base_image[5][5], 1, "image addition should update its region")
        self.assertEqual(base_image[8][8], 0, "image addition should only update its region")

        base_image[6:,6:] += np.ones((4,4))
        
        self.assertEqual(base_image[1][1], 1, "array addition should only update its region")
        self.assertEqual(base_image[6][6], 2, "array addition should update its region")
        self.assertEqual(base_image[8][8], 1, "array addition should update its region")

        
if __name__ == "__main__":
    unittest.main()
        
