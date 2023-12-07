import unittest
from astrophot import image
import astrophot as ap
import torch

from utils import get_astropy_wcs

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

        self.assertTrue(torch.allclose(H.plane_to_world(H.center), torch.ones_like(H.center) * 10), "Center_radec provided, center should be as given in world coordinates")

        # Origin radec
        H = ap.image.Image_Header(
            data_shape = (20,20),
            pixelscale = 0.2,
            origin_radec = (10,10),
        )

        self.assertTrue(torch.allclose(H.plane_to_world(H.origin), torch.ones_like(H.center) * 10), "Origin_radec provided, origin should be as given in world coordinates")

        # Astropy WCS
        wcs = get_astropy_wcs()
        H = ap.image.Image_Header(
            data_shape = (180,180),
            wcs = wcs,
        )

        sky_coord = wcs.pixel_to_world(*wcs.wcs.crpix)
        wcs_world = torch.tensor((sky_coord.ra.deg, sky_coord.dec.deg))
        self.assertTrue(torch.allclose(torch.tensor(wcs.wcs.crpix, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), H.world_to_pixel(wcs_world)), "Astropy WCS initialization should map crval crpix coordinates")
        
        
    def test_image_header_wcs_roundtrip(self):

        wcs = get_astropy_wcs()
        # Minimial input
        H = ap.image.Image_Header(
            data_shape = (20,20),
            zeropoint = 22.5,
            wcs = wcs,
        )

        self.assertTrue(torch.allclose(H.world_to_plane(H.plane_to_world(torch.zeros_like(H.window.reference_radec))), torch.zeros_like(H.window.reference_radec)), "WCS world/plane roundtrip should return input value")
        self.assertTrue(torch.allclose(H.pixel_to_plane(H.plane_to_pixel(torch.zeros_like(H.window.reference_radec))), torch.zeros_like(H.window.reference_radec)), "WCS pixel/plane roundtrip should return input value")
        self.assertTrue(torch.allclose(H.world_to_pixel(H.pixel_to_world(torch.zeros_like(H.window.reference_radec))), torch.zeros_like(H.window.reference_radec), atol = 1e-6), "WCS world/pixel roundtrip should return input value")

        self.assertTrue(torch.allclose(H.pixel_to_plane_delta(H.plane_to_pixel_delta(torch.ones_like(H.window.reference_radec))), torch.ones_like(H.window.reference_radec)), "WCS pixel/plane delta roundtrip should return input value")

        
        
    def test_iamge_header_repr(self):

        wcs = get_astropy_wcs()
        # Minimial input
        H = ap.image.Image_Header(
            data_shape = (20,20),
            zeropoint = 22.5,
            wcs = wcs,
        )

        S = str(H)
        R = repr(H)
        
