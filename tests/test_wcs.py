import unittest
import astrophot as ap
import numpy as np
import torch


class TestWCS(unittest.TestCase):
    def test_wcs_creation(self):

        # Blank startup
        wcs_blank = ap.image.WCS()

        self.assertEqual(wcs_blank.projection, "gnomonic", "Default projection should be Gnomonic")
        self.assertTrue(torch.all(wcs_blank.reference_radec == 0), "default reference world coordinates should be zeros")

        # Provided parameters
        ap.image.WCS._projection = None # reset
        ap.image.WCS._reference_radec = None # reset
        wcs_set = ap.image.WCS(
            projection = "orthographic",
            reference_radec = (90,10),
        )

        self.assertEqual(wcs_set.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(torch.all(wcs_set.reference_radec == torch.tensor((90,10))), "World coordinates should be as provided")
        self.assertNotEqual(wcs_blank.projection, "orthographic", "Not all WCS objects should be updated")
        self.assertFalse(torch.all(wcs_blank.reference_radec == torch.tensor((90,10))), "Not all WCS objects should be updated")
        

    def test_wcs_round_trip(self):

        for projection in ["gnomonic", "orthographic", "steriographic"]:
            print(projection)
            for ref_coords in [(20.3, 79), (120.2,-19), (300, -50), (0,0)]:
                print(ref_coords)
                ap.image.WCS._projection = None # reset
                ap.image.WCS._reference_radec = None # reset
                wcs = ap.image.WCS(
                    projection = projection,
                    reference_radec = ref_coords,
                )

                test_grid_RA, test_grid_DEC = torch.meshgrid(
                    torch.linspace(ref_coords[0] - 10, ref_coords[0] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # RA
                    torch.linspace(ref_coords[1] - 10, ref_coords[1] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # DEC
                    indexing = "xy",
                )

                project_x, project_y = wcs.world_to_plane(
                    test_grid_RA,
                    test_grid_DEC,
                )

                reproject_RA, reproject_DEC = wcs.plane_to_world(
                    project_x,
                    project_y,
                )

                self.assertTrue(torch.allclose(reproject_RA, test_grid_RA), "Round trip RA should map back to itself")
                self.assertTrue(torch.allclose(reproject_DEC, test_grid_DEC), "Round trip DEC should map back to itself")
            
