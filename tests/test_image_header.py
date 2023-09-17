import unittest
from astrophot import image
import astrophot as ap
import torch

######################################################################
# Image_Header Objects
######################################################################

class TestImageHeader(unittest.TestCase):
    def test_image_header_creation(self):

        # Minimial input
        H = ap.image.Image_Header(
            data_shape = (20,20),
            zeropoint = 22.5,
            pixelscale = 0.2,
        )

        self.assertTrue(torch.all(H.origin == 0), "Origin should be assumed zero if not given")

        # Center
        H = ap.image.Image_Header(
            data_shape = (20,20),
            pixelscale = 0.2,
            center = (10,10),
        )

        self.assertTrue(torch.all(H.origin == 8), "Center provided, origin should be adjusted accordingly")

        # Origin
        H = ap.image.Image_Header(
            data_shape = (20,20),
            pixelscale = 0.2,
            origin = (10,10),
        )

        self.assertTrue(torch.all(H.origin == 10), "Origin provided, origin should be as given")

        # Center radec
        H = ap.image.Image_Header(
            data_shape = (20,20),
            pixelscale = 0.2,
            center_radec = (10,10),
        )

        self.assertTrue(torch.allclose(torch.stack(H.plane_to_world(*H.center)), torch.ones_like(H.center) * 10), "Center_radec provided, center should be as given in world coordinates")

        # Origin radec
        H = ap.image.Image_Header(
            data_shape = (20,20),
            pixelscale = 0.2,
            origin_radec = (10,10),
        )

        self.assertTrue(torch.allclose(torch.stack(H.plane_to_world(*H.origin)), torch.ones_like(H.center) * 10), "Origin_radec provided, origin should be as given in world coordinates")

        
