import unittest
import astrophot as ap
import numpy as np
import torch

class TestWPCS(unittest.TestCase):
    def test_wpcs_creation(self):

        # Blank startup
        wcs_blank = ap.image.WPCS()

        self.assertEqual(wcs_blank.projection, "gnomonic", "Default projection should be Gnomonic")
        self.assertTrue(torch.all(wcs_blank.reference_radec == 0), "default reference world coordinates should be zeros")
        self.assertTrue(torch.all(wcs_blank.reference_planexy == 0), "default reference plane coordinates should be zeros")

        # Provided parameters
        wcs_set = ap.image.WPCS(
            projection = "orthographic",
            reference_radec = (90,10),
        )

        self.assertEqual(wcs_set.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(torch.all(wcs_set.reference_radec == torch.tensor((90,10), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "World coordinates should be as provided")
        self.assertNotEqual(wcs_blank.projection, "orthographic", "Not all WCS objects should be updated")
        self.assertFalse(torch.all(wcs_blank.reference_radec == torch.tensor((90,10), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "Not all WCS objects should be updated")

        wcs_set = wcs_set.copy()
        
        self.assertEqual(wcs_set.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(torch.all(wcs_set.reference_radec == torch.tensor((90,10), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "World coordinates should be as provided")
        self.assertNotEqual(wcs_blank.projection, "orthographic", "Not all WCS objects should be updated")
        self.assertFalse(torch.all(wcs_blank.reference_radec == torch.tensor((90,10), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "Not all WCS objects should be updated")

    def test_wpcs_round_trip(self):

        for projection in ["gnomonic", "orthographic", "steriographic"]:
            print(projection)
            for ref_coords in [(20.3, 79), (120.2,-19), (300, -50), (0,0)]:
                print(ref_coords)
                wcs = ap.image.WPCS(
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

    def test_wpcs_errors(self):
        with self.assertRaises(ap.errors.InvalidWCS):
            wcs = ap.image.WPCS(
                projection = "connor",
            )
        
class TestPPCS(unittest.TestCase):

    def test_ppcs_creation(self):
        # Blank startup
        wcs_blank = ap.image.PPCS()

        self.assertTrue(np.all(wcs_blank.pixelscale.detach().cpu().numpy() == np.array([[1.,0.],[0.,1.]])), "Default pixelscale should be 1")
        self.assertTrue(torch.all(wcs_blank.reference_imageij == -0.5), "default reference pixel coordinates should be -0.5")
        self.assertTrue(torch.all(wcs_blank.reference_imagexy == 0.), "default reference plane coordinates should be zeros")

        # Provided parameters
        wcs_set = ap.image.PPCS(
            pixelscale = [[-0.173205, 0.1],[0.15,0.259808]],
            reference_imageij = (5,10),
            reference_imagexy = (0.12, 0.45),
        )

        self.assertTrue(torch.allclose(wcs_set.pixelscale, torch.tensor([[-0.173205, 0.1],[0.15,0.259808]], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "Provided pixelscale should be used")
        self.assertTrue(torch.allclose(wcs_set.reference_imageij, torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "pixel reference coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs_set.reference_imagexy, torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "plane reference coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs_set.plane_to_pixel(torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "plane reference coordinates should map to pixel reference coordinates")
        self.assertTrue(torch.allclose(wcs_set.pixel_to_plane(torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "pixel reference coordinates should map to plane reference coordinates")

        wcs_set = wcs_set.copy()

        self.assertTrue(torch.allclose(wcs_set.pixelscale, torch.tensor([[-0.173205, 0.1],[0.15,0.259808]], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "Provided pixelscale should be used")
        self.assertTrue(torch.allclose(wcs_set.reference_imageij, torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "pixel reference coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs_set.reference_imagexy, torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "plane reference coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs_set.plane_to_pixel(torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "plane reference coordinates should map to pixel reference coordinates")
        self.assertTrue(torch.allclose(wcs_set.pixel_to_plane(torch.tensor((5.,10.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), torch.tensor((0.12,0.45), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "pixel reference coordinates should map to plane reference coordinates")
        

        wcs_set.pixelscale = None
        
    def test_ppcs_round_trip(self):

        for pixelscale in [0.2, [[0.6, 0.],[0., 0.4]], [[-0.173205, 0.1],[0.15,0.259808]]]:
            print(pixelscale)
            for ref_coords in [(20.3, 79), (120.2,-19), (300, -50), (0,0)]:
                print(ref_coords)
                wcs = ap.image.PPCS(
                    pixelscale = pixelscale,
                    reference_imagexy = ref_coords,
                )

                test_grid_x, test_grid_y = torch.meshgrid(
                    torch.linspace(ref_coords[0] - 10, ref_coords[0] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # x
                    torch.linspace(ref_coords[1] - 10, ref_coords[1] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # y
                    indexing = "xy",
                )

                project_i, project_j = wcs.plane_to_pixel(
                    test_grid_x,
                    test_grid_y,
                )

                reproject_x, reproject_y = wcs.pixel_to_plane(
                    project_i,
                    project_j,
                )

                self.assertTrue(torch.allclose(reproject_x, test_grid_x), "Round trip x should map back to itself")
                self.assertTrue(torch.allclose(reproject_y, test_grid_y), "Round trip y should map back to itself")

class TestWCS(unittest.TestCase):
    def test_wcs_creation(self):

        wcs = ap.image.WCS(
            projection = "orthographic",
            pixelscale = [[-0.173205, 0.1],[0.15,0.259808]],
            reference_radec = (120.2,-19),
            reference_imagexy = (33., 123.),
        )

        wcs2 = wcs.copy()

        self.assertEqual(wcs2.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(torch.allclose(wcs2.reference_radec, wcs.reference_radec), "World coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs2.reference_planexy, wcs.reference_planexy), "Plane coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs2.reference_imagexy, wcs.reference_imagexy), "imagexy coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs2.reference_imageij, wcs.reference_imageij), "imageij coordinates should be as provided")
        self.assertTrue(torch.allclose(wcs2.pixelscale, wcs.pixelscale), "pixelscale should be as provided")
        
        
    def test_wcs_roundtrip(self):
        for pixelscale in [0.2, [[0.6, 0.],[0., 0.4]], [[-0.173205, 0.1],[0.15,0.259808]]]:
            print(pixelscale)
            for ref_coords_xy in [(33., 123.), (-430.2,-11), (-97., 5), (0,0)]:
                for projection in ["gnomonic", "orthographic", "steriographic"]:
                    print(projection)
                    for ref_coords_radec in [(20.3, 79), (120.2,-19), (300, -50), (0,0)]:
                        print(ref_coords_radec)
                        wcs = ap.image.WCS(
                            projection = projection,
                            pixelscale = pixelscale,
                            reference_radec = ref_coords_radec,
                            reference_imagexy = ref_coords_xy,
                        )

                        
                        test_grid_RA, test_grid_DEC = torch.meshgrid(
                            torch.linspace(ref_coords_radec[0] - 10, ref_coords_radec[0] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # RA
                            torch.linspace(ref_coords_radec[1] - 10, ref_coords_radec[1] + 10, 10, dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device), # DEC
                            indexing = "xy",
                        )
                        
                        project_i, project_j = wcs.world_to_pixel(
                            test_grid_RA,
                            test_grid_DEC,
                        )
                        
                        reproject_RA, reproject_DEC = wcs.pixel_to_world(
                            project_i,
                            project_j,
                        )
                        
                        self.assertTrue(torch.allclose(reproject_RA, test_grid_RA), "Round trip RA should map back to itself")
                        self.assertTrue(torch.allclose(reproject_DEC, test_grid_DEC), "Round trip DEC should map back to itself")
    

    def test_wcs_state(self):
        wcs = ap.image.WCS(
            projection = "orthographic",
            pixelscale = [[-0.173205, 0.1],[0.15,0.259808]],
            reference_radec = (120.2,-19),
            reference_imagexy = (33., 123.),
        )

        wcs_state = wcs.get_state()

        new_wcs = ap.image.WCS(state = wcs_state)

        self.assertEqual(wcs.projection, new_wcs.projection, "WCS projection should be set by state")
        self.assertTrue(torch.allclose(wcs.pixelscale, torch.tensor([[-0.173205, 0.1],[0.15,0.259808]], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS pixelscale should be set by state")
        self.assertTrue(torch.allclose(wcs.reference_radec, torch.tensor((120.2,-19), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS reference RA DEC should be set by state")
        self.assertTrue(torch.allclose(wcs.reference_imagexy, torch.tensor((33., 123.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS reference image position should be set by state")

        wcs_state = wcs.get_fits_state()

        new_wcs = ap.image.WCS()
        new_wcs.set_fits_state(state = wcs_state)

        self.assertEqual(wcs.projection, new_wcs.projection, "WCS projection should be set by state")
        self.assertTrue(torch.allclose(wcs.pixelscale, torch.tensor([[-0.173205, 0.1],[0.15,0.259808]], dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS pixelscale should be set by state")
        self.assertTrue(torch.allclose(wcs.reference_radec, torch.tensor((120.2,-19), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS reference RA DEC should be set by state")
        self.assertTrue(torch.allclose(wcs.reference_imagexy, torch.tensor((33., 123.), dtype = ap.AP_config.ap_dtype, device = ap.AP_config.ap_device)), "WCS reference image position should be set by state")
        
    def test_wcs_repr(self):

        wcs = ap.image.WCS(
            projection = "orthographic",
            pixelscale = [[-0.173205, 0.1],[0.15,0.259808]],
            reference_radec = (120.2,-19),
            reference_imagexy = (33., 123.),
        )

        S = str(wcs)
        R = repr(wcs)
