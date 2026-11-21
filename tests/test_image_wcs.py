"""Pin the coordinates of AstroPhot images against astropy.

The images here are deliberately not square, with the reference pixel away
from the centre, so that any mix up of the two axes shows up.
"""

import astrophot as ap
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

NX, NY = 60, 40  # FITS NAXIS1, NAXIS2. The numpy data shape is (NY, NX)
SCALE = 0.2 / 3600  # pixel size in degrees
THETA = np.deg2rad(30)

# Everything except the image size, CTYPE and data
WCS_HEADERS = {
    "centred": {
        "CRVAL1": 195.0588,
        "CRVAL2": 28.0608,
        "CRPIX1": 30.5,
        "CRPIX2": 20.5,
        "CD1_1": -SCALE,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": SCALE,
    },
    "offcentre": {
        "CRVAL1": 195.0588,
        "CRVAL2": 28.0608,
        "CRPIX1": 11.0,
        "CRPIX2": 33.25,
        "CD1_1": -SCALE,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": SCALE,
    },
    "rotated": {
        "CRVAL1": 10.5,
        "CRVAL2": -43.2,
        "CRPIX1": 48.0,
        "CRPIX2": -5.0,
        "CD1_1": -SCALE * np.cos(THETA),
        "CD1_2": SCALE * np.sin(THETA),
        "CD2_1": SCALE * np.sin(THETA),
        "CD2_2": SCALE * np.cos(THETA),
    },
    "not_flipped_rectangular_pixels": {
        "CRVAL1": 359.999,
        "CRVAL2": 0.001,
        "CRPIX1": 7.5,
        "CRPIX2": 31.0,
        "CD1_1": 1.5 * SCALE,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": SCALE,
    },
    "cdelt_pc": {
        "CRVAL1": 150.1,
        "CRVAL2": 82.0,
        "CRPIX1": 21.0,
        "CRPIX2": 9.5,
        "CDELT1": -SCALE,
        "CDELT2": 2 * SCALE,
        "PC1_1": np.cos(THETA),
        "PC1_2": -np.sin(THETA),
        "PC2_1": np.sin(THETA),
        "PC2_2": np.cos(THETA),
    },
}


def make_header(name):
    return {
        "NAXIS": 2,
        "NAXIS1": NX,
        "NAXIS2": NY,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        **WCS_HEADERS[name],
    }


@pytest.fixture(params=list(WCS_HEADERS))
def astropy_wcs(request):
    return WCS(make_header(request.param))


def sample_pixels(nx=NX, ny=NY):
    """Pixel centres over the whole image (corners included) and some fractional positions."""
    x, y = np.meshgrid(np.linspace(0, nx - 1, 7), np.linspace(0, ny - 1, 5))
    x = np.concatenate((x.flatten(), [-0.5, 12.3, nx - 0.5]))
    y = np.concatenate((y.flatten(), [-0.5, 27.9, ny - 0.5]))
    return x, y


def as_array(x):
    return ap.backend.as_array(x, dtype=ap.config.DTYPE, device=ap.config.DEVICE)


def image_pixel_to_world(image, x, y):
    ra, dec = image.pixel_to_world(as_array(x), as_array(y))
    return ap.backend.to_numpy(ra), ap.backend.to_numpy(dec)


def separation_arcsec(ra1, dec1, ra2, dec2):
    dra = (ra1 - ra2 + 180) % 360 - 180
    return 3600 * np.hypot(dra * np.cos(np.deg2rad(dec2)), dec1 - dec2)


def assert_same_sky(image, wcs, x, y, xw=None, yw=None, tol=1e-6):
    """Pixel (x, y) of the image and pixel (xw, yw) of the wcs are the same place on the sky."""
    ra, dec = image_pixel_to_world(image, x, y)
    ra_ref, dec_ref = wcs.all_pix2world(x if xw is None else xw, y if yw is None else yw, 0)
    sep = separation_arcsec(ra, dec, ra_ref, dec_ref)
    assert np.all(sep < tol), f"AstroPhot is up to {sep.max():.3g} arcsec from astropy"


def test_wcs_pixel_to_world_matches_astropy(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    assert_same_sky(image, astropy_wcs, *sample_pixels())


def test_wcs_world_to_pixel_matches_astropy(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    x, y = sample_pixels()
    ra, dec = astropy_wcs.all_pix2world(x, y, 0)
    i, j = image.world_to_pixel(as_array(ra), as_array(dec))
    assert np.allclose(ap.backend.to_numpy(i), x, atol=1e-6), "i should be the FITS axis 1 pixel"
    assert np.allclose(ap.backend.to_numpy(j), y, atol=1e-6), "j should be the FITS axis 2 pixel"


def test_wcs_reference_values(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    assert image._data.shape == (NX, NY), "internal data should be indexed (FITS axis 1, axis 2)"
    assert np.allclose(
        image.crpix, astropy_wcs.wcs.crpix - 1
    ), "crpix should be the 0-indexed FITS reference pixel, in FITS axis order"
    assert np.allclose(image.crval.npvalue, astropy_wcs.wcs.crval), "crval should be FITS CRVAL"
    assert np.allclose(
        image.CD.npvalue, 3600 * astropy_wcs.pixel_scale_matrix
    ), "CD should be the FITS CD matrix in arcsec"


def test_wcs_data_orientation(astropy_wcs):
    """The pixel values are found at the sky position astropy gives for them."""
    x0, y0 = 47, 6
    data = np.zeros((NY, NX))
    data[y0, x0] = 1.0  # numpy indexing is [FITS axis 2, FITS axis 1]
    image = ap.TargetImage(data=data, wcs=astropy_wcs)
    assert image.data[y0, x0].item() == 1.0, "data should keep the numpy indexing it was given"
    assert image._data[x0, y0].item() == 1.0, "internal data should be indexed (i, j)"

    # sky position of the bright pixel, from a meshgrid of the whole image
    i, j = image.pixel_center_meshgrid()
    ra, dec = image.pixel_to_world(i, j)
    bright = ap.backend.to_numpy(image._data) == 1.0
    ra_ref, dec_ref = astropy_wcs.all_pix2world(x0, y0, 0)
    sep = separation_arcsec(
        ap.backend.to_numpy(ra)[bright], ap.backend.to_numpy(dec)[bright], ra_ref, dec_ref
    )
    assert np.all(sep < 1e-6), "bright pixel should be where astropy puts it on the sky"


def test_wcs_window_matches_astropy_slice(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    x_low, x_high, y_low, y_high = 13, 55, 4, 22
    cutout = image[ap.Window((x_low, x_high, y_low, y_high), image)]
    assert cutout._data.shape == (x_high - x_low, y_high - y_low)

    x, y = sample_pixels(x_high - x_low, y_high - y_low)
    # same pixels on the full image
    assert_same_sky(cutout, astropy_wcs, x, y, x + x_low, y + y_low)
    # astropy slices in numpy order
    assert_same_sky(cutout, astropy_wcs[y_low:y_high, x_low:x_high], x, y)


def test_wcs_reduce_matches_astropy(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    reduced = image.reduce(2)
    x, y = sample_pixels(NX // 2, NY // 2)
    # centre of reduced pixel n is the middle of original pixels 2n and 2n + 1
    assert_same_sky(reduced, astropy_wcs, x, y, 2 * x + 0.5, 2 * y + 0.5)


def test_wcs_save_load_matches_astropy(astropy_wcs, tmp_path):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    image.save(tmp_path / "image.fits")
    x, y = sample_pixels()

    # the header that AstroPhot writes, as astropy reads it
    with fits.open(tmp_path / "image.fits") as hdul:
        assert hdul[0].data.shape == (NY, NX)
        saved_wcs = WCS(hdul[0].header)
    assert_same_sky(image, saved_wcs, x, y)

    loaded = ap.TargetImage(filename=tmp_path / "image.fits")
    assert_same_sky(loaded, astropy_wcs, x, y)


def test_wcs_load_fits_not_written_by_astrophot(astropy_wcs, tmp_path):
    hdu = fits.PrimaryHDU(np.zeros((NY, NX)), header=astropy_wcs.to_header())
    hdu.writeto(tmp_path / "other.fits")
    assert "CD1_1" not in hdu.header, "astropy should write this header with CDELT and PC"

    loaded = ap.Image(filename=tmp_path / "other.fits")
    assert_same_sky(loaded, astropy_wcs, *sample_pixels())


def test_get_astropywcs(astropy_wcs):
    image = ap.TargetImage(data=np.zeros((NY, NX)), wcs=astropy_wcs)
    wcs = image.get_astropywcs()
    assert wcs.pixel_shape == (NX, NY)
    x, y = sample_pixels()
    assert_same_sky(image, wcs, x, y)
    assert np.allclose(wcs.all_pix2world(x, y, 0), astropy_wcs.all_pix2world(x, y, 0))


def test_get_astropywcs_with_crtan():
    """crtan shifts the tangent plane, astropy has to give the same sky positions."""
    image = ap.Image(
        data=np.zeros((NY, NX)),
        crpix=(10.0, 33.0),
        crtan=(1.5, -4.0),
        crval=(160.0, 45.0),
        CD=0.25 * np.array([[-np.cos(THETA), np.sin(THETA)], [np.sin(THETA), np.cos(THETA)]]),
    )
    assert_same_sky(image, image.get_astropywcs(), *sample_pixels())


def test_sip_matches_astropy():
    header = make_header("offcentre")
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    sip_a = {(2, 0): 2e-5, (1, 1): -1e-5, (0, 2): 3e-5, (0, 3): 4e-7}
    sip_b = {(2, 0): -1e-5, (1, 1): 2.5e-5, (0, 2): 1e-5, (3, 0): -3e-7}
    header["A_ORDER"] = 3
    header["B_ORDER"] = 3
    for (p, q), value in sip_a.items():
        header[f"A_{p}_{q}"] = value
    for (p, q), value in sip_b.items():
        header[f"B_{p}_{q}"] = value
    wcs = WCS(header)

    image = ap.SIPTargetImage(data=np.zeros((NY, NX)), wcs=wcs, sipA=sip_a, sipB=sip_b)
    # the distortion is stored at the pixel centres, so compare there
    x, y = np.meshgrid(np.arange(0.0, NX, 7), np.arange(0.0, NY, 5))
    x, y = x.flatten(), y.flatten()
    ra_linear, dec_linear = wcs.wcs_pix2world(x, y, 0)
    ra, dec = wcs.all_pix2world(x, y, 0)
    assert separation_arcsec(ra, dec, ra_linear, dec_linear).max() > 1e-2, "test needs distortion"
    assert_same_sky(image, wcs, x, y)


def test_wcs_roundtrip():
    """Each coordinate transform is undone by its inverse, with a crtan offset."""
    I = ap.Image(
        data=np.zeros((21, 21)),
        zeropoint=22.5,
        crpix=(10, 10),
        crtan=(1.0, -10.0),
        crval=(160.0, 45.0),
        CD=0.05
        * np.array(
            [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]]
        ),
    )

    assert ap.backend.allclose(
        ap.backend.stack(I.world_to_plane(*I.plane_to_world(*I.center))),
        I.center,
    ), "WCS world/plane roundtrip should return input value"
    assert ap.backend.allclose(
        ap.backend.stack(I.pixel_to_plane(*I.plane_to_pixel(*I.center))),
        I.center,
    ), "WCS pixel/plane roundtrip should return input value"
    assert ap.backend.allclose(
        ap.backend.stack(I.world_to_pixel(*I.pixel_to_world(*ap.backend.zeros_like(I.center)))),
        ap.backend.zeros_like(I.center),
        atol=1e-6,
    ), "WCS world/pixel roundtrip should return input value"
