import unittest
import astrophot as ap
import numpy as np
import torch

import astropy.wcs
from astropy.coordinates import SkyCoord
import astropy.units as u


class TestWPCS(unittest.TestCase):
    def test_wpcs_creation(self):

        # Blank startup
        wcs_blank = ap.image.WPCS()

        self.assertEqual(wcs_blank.projection, "gnomonic", "Default projection should be Gnomonic")
        self.assertTrue(
            torch.all(wcs_blank.reference_radec == 0),
            "default reference world coordinates should be zeros",
        )
        self.assertTrue(
            torch.all(wcs_blank.reference_planexy == 0),
            "default reference plane coordinates should be zeros",
        )

        # Provided parameters
        wcs_set = ap.image.WPCS(
            projection="orthographic",
            reference_radec=(90, 10),
        )

        self.assertEqual(wcs_set.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(
            torch.all(
                wcs_set.reference_radec
                == torch.tensor(
                    (90, 10), dtype=ap.AP_config.ap_dtype, device=ap.AP_config.ap_device
                )
            ),
            "World coordinates should be as provided",
        )
        self.assertNotEqual(
            wcs_blank.projection,
            "orthographic",
            "Not all WCS objects should be updated",
        )
        self.assertFalse(
            torch.all(
                wcs_blank.reference_radec
                == torch.tensor(
                    (90, 10), dtype=ap.AP_config.ap_dtype, device=ap.AP_config.ap_device
                )
            ),
            "Not all WCS objects should be updated",
        )

        wcs_set = wcs_set.copy()

        self.assertEqual(wcs_set.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(
            torch.all(
                wcs_set.reference_radec
                == torch.tensor(
                    (90, 10), dtype=ap.AP_config.ap_dtype, device=ap.AP_config.ap_device
                )
            ),
            "World coordinates should be as provided",
        )
        self.assertNotEqual(
            wcs_blank.projection,
            "orthographic",
            "Not all WCS objects should be updated",
        )
        self.assertFalse(
            torch.all(
                wcs_blank.reference_radec
                == torch.tensor(
                    (90, 10), dtype=ap.AP_config.ap_dtype, device=ap.AP_config.ap_device
                )
            ),
            "Not all WCS objects should be updated",
        )

    def test_wpcs_round_trip(self):

        for projection in ["gnomonic", "orthographic", "steriographic"]:
            print(projection)
            for ref_coords in [(20.3, 79), (120.2, -19), (300, -50), (0, 0)]:
                print(ref_coords)
                wcs = ap.image.WPCS(
                    projection=projection,
                    reference_radec=ref_coords,
                )

                test_grid_RA, test_grid_DEC = torch.meshgrid(
                    torch.linspace(
                        ref_coords[0] - 10,
                        ref_coords[0] + 10,
                        10,
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    ),  # RA
                    torch.linspace(
                        ref_coords[1] - 10,
                        ref_coords[1] + 10,
                        10,
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    ),  # DEC
                    indexing="xy",
                )

                project_x, project_y = wcs.world_to_plane(
                    test_grid_RA,
                    test_grid_DEC,
                )

                reproject_RA, reproject_DEC = wcs.plane_to_world(
                    project_x,
                    project_y,
                )

                self.assertTrue(
                    torch.allclose(reproject_RA, test_grid_RA),
                    "Round trip RA should map back to itself",
                )
                self.assertTrue(
                    torch.allclose(reproject_DEC, test_grid_DEC),
                    "Round trip DEC should map back to itself",
                )

    def test_wpcs_errors(self):
        with self.assertRaises(ap.errors.InvalidWCS):
            wcs = ap.image.WPCS(
                projection="connor",
            )


class TestPPCS(unittest.TestCase):

    def test_ppcs_creation(self):
        # Blank startup
        wcs_blank = ap.image.PPCS()

        self.assertTrue(
            np.all(
                wcs_blank.pixelscale.detach().cpu().numpy() == np.array([[1.0, 0.0], [0.0, 1.0]])
            ),
            "Default pixelscale should be 1",
        )
        self.assertTrue(
            torch.all(wcs_blank.reference_imageij == -0.5),
            "default reference pixel coordinates should be -0.5",
        )
        self.assertTrue(
            torch.all(wcs_blank.reference_imagexy == 0.0),
            "default reference plane coordinates should be zeros",
        )

        # Provided parameters
        wcs_set = ap.image.PPCS(
            pixelscale=[[-0.173205, 0.1], [0.15, 0.259808]],
            reference_imageij=(5, 10),
            reference_imagexy=(0.12, 0.45),
        )

        self.assertTrue(
            torch.allclose(
                wcs_set.pixelscale,
                torch.tensor(
                    [[-0.173205, 0.1], [0.15, 0.259808]],
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "Provided pixelscale should be used",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.reference_imageij,
                torch.tensor(
                    (5.0, 10.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "pixel reference coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.reference_imagexy,
                torch.tensor(
                    (0.12, 0.45),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "plane reference coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.plane_to_pixel(
                    torch.tensor(
                        (0.12, 0.45),
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    )
                ),
                torch.tensor(
                    (5.0, 10.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "plane reference coordinates should map to pixel reference coordinates",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.pixel_to_plane(
                    torch.tensor(
                        (5.0, 10.0),
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    )
                ),
                torch.tensor(
                    (0.12, 0.45),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "pixel reference coordinates should map to plane reference coordinates",
        )

        wcs_set = wcs_set.copy()

        self.assertTrue(
            torch.allclose(
                wcs_set.pixelscale,
                torch.tensor(
                    [[-0.173205, 0.1], [0.15, 0.259808]],
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "Provided pixelscale should be used",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.reference_imageij,
                torch.tensor(
                    (5.0, 10.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "pixel reference coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.reference_imagexy,
                torch.tensor(
                    (0.12, 0.45),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "plane reference coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.plane_to_pixel(
                    torch.tensor(
                        (0.12, 0.45),
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    )
                ),
                torch.tensor(
                    (5.0, 10.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "plane reference coordinates should map to pixel reference coordinates",
        )
        self.assertTrue(
            torch.allclose(
                wcs_set.pixel_to_plane(
                    torch.tensor(
                        (5.0, 10.0),
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    )
                ),
                torch.tensor(
                    (0.12, 0.45),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "pixel reference coordinates should map to plane reference coordinates",
        )

        wcs_set.pixelscale = None

    def test_ppcs_round_trip(self):

        for pixelscale in [
            0.2,
            [[0.6, 0.0], [0.0, 0.4]],
            [[-0.173205, 0.1], [0.15, 0.259808]],
        ]:
            print(pixelscale)
            for ref_coords in [(20.3, 79), (120.2, -19), (300, -50), (0, 0)]:
                print(ref_coords)
                wcs = ap.image.PPCS(
                    pixelscale=pixelscale,
                    reference_imagexy=ref_coords,
                )

                test_grid_x, test_grid_y = torch.meshgrid(
                    torch.linspace(
                        ref_coords[0] - 10,
                        ref_coords[0] + 10,
                        10,
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    ),  # x
                    torch.linspace(
                        ref_coords[1] - 10,
                        ref_coords[1] + 10,
                        10,
                        dtype=ap.AP_config.ap_dtype,
                        device=ap.AP_config.ap_device,
                    ),  # y
                    indexing="xy",
                )

                project_i, project_j = wcs.plane_to_pixel(
                    test_grid_x,
                    test_grid_y,
                )

                reproject_x, reproject_y = wcs.pixel_to_plane(
                    project_i,
                    project_j,
                )

                self.assertTrue(
                    torch.allclose(reproject_x, test_grid_x),
                    "Round trip x should map back to itself",
                )
                self.assertTrue(
                    torch.allclose(reproject_y, test_grid_y),
                    "Round trip y should map back to itself",
                )



class TestWCS(unittest.TestCase):
    def test_wcs_creation(self):

        wcs = ap.image.WCS(
            projection="orthographic",
            pixelscale=[[-0.173205, 0.1], [0.15, 0.259808]],
            reference_radec=(120.2, -19),
            reference_imagexy=(33.0, 123.0),
        )

        wcs2 = wcs.copy()

        self.assertEqual(wcs2.projection, "orthographic", "Provided projection was Orthographic")
        self.assertTrue(
            torch.allclose(wcs2.reference_radec, wcs.reference_radec),
            "World coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(wcs2.reference_planexy, wcs.reference_planexy),
            "Plane coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(wcs2.reference_imagexy, wcs.reference_imagexy),
            "imagexy coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(wcs2.reference_imageij, wcs.reference_imageij),
            "imageij coordinates should be as provided",
        )
        self.assertTrue(
            torch.allclose(wcs2.pixelscale, wcs.pixelscale),
            "pixelscale should be as provided",
        )

    def test_wcs_roundtrip(self):
        for pixelscale in [
            0.2,
            [[0.6, 0.0], [0.0, 0.4]],
            [[-0.173205, 0.1], [0.15, 0.259808]],
        ]:
            print(pixelscale)
            for ref_coords_xy in [(33.0, 123.0), (-430.2, -11), (-97.0, 5), (0, 0)]:
                for projection in ["gnomonic", "orthographic", "steriographic"]:
                    print(projection)
                    for ref_coords_radec in [
                        (20.3, 79),
                        (120.2, -19),
                        (300, -50),
                        (0, 0),
                    ]:
                        print(ref_coords_radec)
                        wcs = ap.image.WCS(
                            projection=projection,
                            pixelscale=pixelscale,
                            reference_radec=ref_coords_radec,
                            reference_imagexy=ref_coords_xy,
                        )

                        test_grid_RA, test_grid_DEC = torch.meshgrid(
                            torch.linspace(
                                ref_coords_radec[0] - 10,
                                ref_coords_radec[0] + 10,
                                10,
                                dtype=ap.AP_config.ap_dtype,
                                device=ap.AP_config.ap_device,
                            ),  # RA
                            torch.linspace(
                                ref_coords_radec[1] - 10,
                                ref_coords_radec[1] + 10,
                                10,
                                dtype=ap.AP_config.ap_dtype,
                                device=ap.AP_config.ap_device,
                            ),  # DEC
                            indexing="xy",
                        )

                        project_i, project_j = wcs.world_to_pixel(
                            test_grid_RA,
                            test_grid_DEC,
                        )

                        reproject_RA, reproject_DEC = wcs.pixel_to_world(
                            project_i,
                            project_j,
                        )

                        self.assertTrue(
                            torch.allclose(reproject_RA, test_grid_RA),
                            "Round trip RA should map back to itself",
                        )
                        self.assertTrue(
                            torch.allclose(reproject_DEC, test_grid_DEC),
                            "Round trip DEC should map back to itself",
                        )

    def test_wcs_state(self):
        wcs = ap.image.WCS(
            projection="orthographic",
            pixelscale=[[-0.173205, 0.1], [0.15, 0.259808]],
            reference_radec=(120.2, -19),
            reference_imagexy=(33.0, 123.0),
        )

        wcs_state = wcs.get_state()

        new_wcs = ap.image.WCS(state=wcs_state)

        self.assertEqual(
            wcs.projection, new_wcs.projection, "WCS projection should be set by state"
        )
        self.assertTrue(
            torch.allclose(
                wcs.pixelscale,
                torch.tensor(
                    [[-0.173205, 0.1], [0.15, 0.259808]],
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS pixelscale should be set by state",
        )
        self.assertTrue(
            torch.allclose(
                wcs.reference_radec,
                torch.tensor(
                    (120.2, -19),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS reference RA DEC should be set by state",
        )
        self.assertTrue(
            torch.allclose(
                wcs.reference_imagexy,
                torch.tensor(
                    (33.0, 123.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS reference image position should be set by state",
        )

        wcs_state = wcs.get_fits_state()

        new_wcs = ap.image.WCS()
        new_wcs.set_fits_state(state=wcs_state)

        self.assertEqual(
            wcs.projection, new_wcs.projection, "WCS projection should be set by state"
        )
        self.assertTrue(
            torch.allclose(
                wcs.pixelscale,
                torch.tensor(
                    [[-0.173205, 0.1], [0.15, 0.259808]],
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS pixelscale should be set by state",
        )
        self.assertTrue(
            torch.allclose(
                wcs.reference_radec,
                torch.tensor(
                    (120.2, -19),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS reference RA DEC should be set by state",
        )
        self.assertTrue(
            torch.allclose(
                wcs.reference_imagexy,
                torch.tensor(
                    (33.0, 123.0),
                    dtype=ap.AP_config.ap_dtype,
                    device=ap.AP_config.ap_device,
                ),
            ),
            "WCS reference image position should be set by state",
        )

    def test_wcs_repr(self):

        wcs = ap.image.WCS(
            projection="orthographic",
            pixelscale=[[-0.173205, 0.1], [0.15, 0.259808]],
            reference_radec=(120.2, -19),
            reference_imagexy=(33.0, 123.0),
        )

        S = str(wcs)
        R = repr(wcs)

    def test_wcs_sip_loads(self):
        # Sample WCS with SIP from a LSST DESC DC2 image
        wcs_header_string_sip = "XTENSION= 'IMAGE   '           / binary table extension                         BITPIX  =                  -32 / data type of original image                    NAXIS   =                    2 / dimension of original image                    NAXIS1  =                 4072 / length of original image axis                  NAXIS2  =                 4000 / length of original image axis                  PCOUNT  =                    0 / size of special data area                      GCOUNT  =                    1 / one data group (required keyword)              WCSAXES =                    2 / Number of WCS axes                             CRPIX1  =          2283.298079 / Reference pixel on axis 1                      CRPIX2  =          1983.106972 / Reference pixel on axis 2                      CRVAL1  =     60.1674408822151 / Value at ref. pixel on axis 1                  CRVAL2  =    -44.0240070609999 / Value at ref. pixel on axis 2                  CTYPE1  = 'RA---TAN-SIP'       / Type of co-ordinate on axis 1                  CTYPE2  = 'DEC--TAN-SIP'       / Type of co-ordinate on axis 2                  CD1_1   = -5.20109992868063E-05 / Transformation matrix element                 CD1_2   = 1.94997314765446E-05 / Transformation matrix element                  CD2_1   = 1.94757706702473E-05 / Transformation matrix element                  CD2_2   = 5.19541953270649E-05 / Transformation matrix element                  RADESYS = 'ICRS    '           / Reference frame for RA/DEC values              A_0_2   = -3.91349613255531E-09 / SIP forward distortion coeff                  A_1_1   = 5.33854528013099E-08 / SIP forward distortion coeff                   A_2_0   = 3.51409551569656E-08 / SIP forward distortion coeff                   A_ORDER =                    2 / SIP max order                                  B_0_2   = 8.12233002801473E-08 / SIP forward distortion coeff                   B_1_1   = 3.44720358757694E-08 / SIP forward distortion coeff                   B_2_0   = 2.30115438387554E-08 / SIP forward distortion coeff                   B_ORDER =                    2 / SIP max order                                  AP_0_0  =  8.7791082937263E-09 / SIP inverse distortion coeff                   AP_0_1  = -2.17912410330855E-12 / SIP inverse distortion coeff                  AP_0_2  = 3.91349357784246E-09 / SIP inverse distortion coeff                   AP_0_3  = -8.44387640365176E-16 / SIP inverse distortion coeff                  AP_1_0  = -2.68562949656825E-12 / SIP inverse distortion coeff                  AP_1_1  = -5.33854625062645E-08 / SIP inverse distortion coeff                  AP_1_2  = 6.64198754491308E-15 / SIP inverse distortion coeff                   AP_2_0  = -3.51409612714431E-08 / SIP inverse distortion coeff                  AP_2_1  =  7.2891568050057E-15 / SIP inverse distortion coeff                   AP_3_0  = 3.69866703175515E-15 / SIP inverse distortion coeff                   AP_ORDER=                    3 / SIP inverse max order                          BP_0_0  = 2.43008121697749E-08 / SIP inverse distortion coeff                   BP_0_1  = -3.09252623509337E-12 / SIP inverse distortion coeff                  BP_0_2  = -8.12233210883521E-08 / SIP inverse distortion coeff                  BP_0_3  =  1.3060003652358E-14 / SIP inverse distortion coeff                   BP_1_0  = -3.52237948045827E-12 / SIP inverse distortion coeff                  BP_1_1  = -3.44720526297139E-08 / SIP inverse distortion coeff                  BP_1_2  = 1.00611796026085E-14 / SIP inverse distortion coeff                   BP_2_0  = -2.30115520819468E-08 / SIP inverse distortion coeff                  BP_2_1  = 8.59570883767581E-15 / SIP inverse distortion coeff                   BP_3_0  = 2.41084920715199E-15 / SIP inverse distortion coeff                   BP_ORDER=                    3 / SIP inverse max order                          LTV1    =                   0.                                                  LTV2    =                   0.                                                  INHERIT =                    T                                                  EXTTYPE = 'IMAGE   '                                                            EXTNAME = 'IMAGE   '                                                            CTYPE1A = 'LINEAR  '           / Type of projection                             CTYPE2A = 'LINEAR  '           / Type of projection                             CRPIX1A =                   1. / Column Pixel Coordinate of Reference           CRPIX2A =                   1. / Row Pixel Coordinate of Reference              CRVAL1A =                   0. / Column pixel of Reference Pixel                CRVAL2A =                   0. / Row pixel of Reference Pixel                   CUNIT1A = 'PIXEL   '           / Column unit                                    CUNIT2A = 'PIXEL   '           / Row unit                                       BZERO   =   0.000000000000E+00 / Scaling: MEMORY = BZERO + BSCALE * DISK        BSCALE  =   1.000000000000E+00 / Scaling: MEMORY = BZERO + BSCALE * DISK        END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             "
        astropy_wcs_sip = astropy.wcs.WCS(wcs_header_string_sip)
        ap_wcs_sip = ap.image.wcs.WCS(wcs=astropy_wcs_sip)  # wcs_header_string_sip)
        self.assertTrue(ap_wcs_sip is not None)

    def skip_test_wcs_sip_matches_astropy(self):
        # Sample WCS with SIP from a LSST DESC DC2 image
        wcs_header_string_sip = "XTENSION= 'IMAGE   '           / binary table extension                         BITPIX  =                  -32 / data type of original image                    NAXIS   =                    2 / dimension of original image                    NAXIS1  =                 4072 / length of original image axis                  NAXIS2  =                 4000 / length of original image axis                  PCOUNT  =                    0 / size of special data area                      GCOUNT  =                    1 / one data group (required keyword)              WCSAXES =                    2 / Number of WCS axes                             CRPIX1  =          2283.298079 / Reference pixel on axis 1                      CRPIX2  =          1983.106972 / Reference pixel on axis 2                      CRVAL1  =     60.1674408822151 / Value at ref. pixel on axis 1                  CRVAL2  =    -44.0240070609999 / Value at ref. pixel on axis 2                  CTYPE1  = 'RA---TAN-SIP'       / Type of co-ordinate on axis 1                  CTYPE2  = 'DEC--TAN-SIP'       / Type of co-ordinate on axis 2                  CD1_1   = -5.20109992868063E-05 / Transformation matrix element                 CD1_2   = 1.94997314765446E-05 / Transformation matrix element                  CD2_1   = 1.94757706702473E-05 / Transformation matrix element                  CD2_2   = 5.19541953270649E-05 / Transformation matrix element                  RADESYS = 'ICRS    '           / Reference frame for RA/DEC values              A_0_2   = -3.91349613255531E-09 / SIP forward distortion coeff                  A_1_1   = 5.33854528013099E-08 / SIP forward distortion coeff                   A_2_0   = 3.51409551569656E-08 / SIP forward distortion coeff                   A_ORDER =                    2 / SIP max order                                  B_0_2   = 8.12233002801473E-08 / SIP forward distortion coeff                   B_1_1   = 3.44720358757694E-08 / SIP forward distortion coeff                   B_2_0   = 2.30115438387554E-08 / SIP forward distortion coeff                   B_ORDER =                    2 / SIP max order                                  AP_0_0  =  8.7791082937263E-09 / SIP inverse distortion coeff                   AP_0_1  = -2.17912410330855E-12 / SIP inverse distortion coeff                  AP_0_2  = 3.91349357784246E-09 / SIP inverse distortion coeff                   AP_0_3  = -8.44387640365176E-16 / SIP inverse distortion coeff                  AP_1_0  = -2.68562949656825E-12 / SIP inverse distortion coeff                  AP_1_1  = -5.33854625062645E-08 / SIP inverse distortion coeff                  AP_1_2  = 6.64198754491308E-15 / SIP inverse distortion coeff                   AP_2_0  = -3.51409612714431E-08 / SIP inverse distortion coeff                  AP_2_1  =  7.2891568050057E-15 / SIP inverse distortion coeff                   AP_3_0  = 3.69866703175515E-15 / SIP inverse distortion coeff                   AP_ORDER=                    3 / SIP inverse max order                          BP_0_0  = 2.43008121697749E-08 / SIP inverse distortion coeff                   BP_0_1  = -3.09252623509337E-12 / SIP inverse distortion coeff                  BP_0_2  = -8.12233210883521E-08 / SIP inverse distortion coeff                  BP_0_3  =  1.3060003652358E-14 / SIP inverse distortion coeff                   BP_1_0  = -3.52237948045827E-12 / SIP inverse distortion coeff                  BP_1_1  = -3.44720526297139E-08 / SIP inverse distortion coeff                  BP_1_2  = 1.00611796026085E-14 / SIP inverse distortion coeff                   BP_2_0  = -2.30115520819468E-08 / SIP inverse distortion coeff                  BP_2_1  = 8.59570883767581E-15 / SIP inverse distortion coeff                   BP_3_0  = 2.41084920715199E-15 / SIP inverse distortion coeff                   BP_ORDER=                    3 / SIP inverse max order                          LTV1    =                   0.                                                  LTV2    =                   0.                                                  INHERIT =                    T                                                  EXTTYPE = 'IMAGE   '                                                            EXTNAME = 'IMAGE   '                                                            CTYPE1A = 'LINEAR  '           / Type of projection                             CTYPE2A = 'LINEAR  '           / Type of projection                             CRPIX1A =                   1. / Column Pixel Coordinate of Reference           CRPIX2A =                   1. / Row Pixel Coordinate of Reference              CRVAL1A =                   0. / Column pixel of Reference Pixel                CRVAL2A =                   0. / Row pixel of Reference Pixel                   CUNIT1A = 'PIXEL   '           / Column unit                                    CUNIT2A = 'PIXEL   '           / Row unit                                       BZERO   =   0.000000000000E+00 / Scaling: MEMORY = BZERO + BSCALE * DISK        BSCALE  =   1.000000000000E+00 / Scaling: MEMORY = BZERO + BSCALE * DISK        END                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             "
        astropy_wcs_sip = astropy.wcs.WCS(wcs_header_string_sip)
        ap_wcs_sip = ap.image.wcs.WCS(wcs=astropy_wcs_sip)  # wcs_header_string_sip)

        # A set of matched coordinates from ds9 'c' coordinate inspection
        (ra, dec, x, y) = (60.206024, -44.036378, 1736.391387, 1948.498631)

        # AstroPy should pass
        astropy_test_ra_dec = astropy_wcs_sip.pixel_to_world(x, y)
        np.testing.assert_allclose(
            [astropy_test_ra_dec.ra.degree, astropy_test_ra_dec.dec.degree], [ra, dec]
        )
        astropy_test_x_y = astropy_wcs_sip.world_to_pixel(SkyCoord(ra, dec, unit=u.deg))
        np.testing.assert_allclose(astropy_test_x_y, [x, y])

        # AstroPhot WCS should fail because SIP isn't implemented
        ap_test_ra_dec = ap_wcs_sip.pixel_to_world(torch.tensor([x]), torch.tensor([y]))
        np.testing.assert_array_equal(ap_test_ra_dec[0], ra)
        np.testing.assert_array_equal(ap_test_ra_dec[1], dec)

        ap_test_x_y = ap_wcs_sip.world_to_pixel(np.asarray(ra), np.asarray(dec))
        np.testing.assert_array_equal(ap_test_x_y[0], x)
        np.testing.assert_array_equal(ap_test_x_y[1], y)
