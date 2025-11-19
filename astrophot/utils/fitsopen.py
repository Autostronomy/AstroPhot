import numpy as np
import warnings
from astropy.utils.data import download_file
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from numpy.core.defchararray import startswith

try:
    from pyvo.dal import sia
except:
    sia = None
import os

# Suppress common Astropy warnings that can clutter CI logs
warnings.simplefilter("ignore", category=AstropyWarning)


def flip_hdu(hdu):
    """
    Flips the image data in the FITS HDU on the RA axis to match the expected orientation.

    Args:
        hdu (astropy.io.fits.HDUList): The FITS HDU to be flipped.

    Returns:
        astropy.io.fits.HDUList: The flipped FITS HDU.
    """
    assert "CD1_1" in hdu[0].header, "HDU does not contain WCS information."
    assert "CD2_1" in hdu[0].header, "HDU does not contain WCS information."
    assert "CRPIX1" in hdu[0].header, "HDU does not contain WCS information."
    assert "NAXIS1" in hdu[0].header, "HDU does not contain WCS information."
    hdu[0].data = hdu[0].data[:, ::-1].copy()
    hdu[0].header["CD1_1"] = -hdu[0].header["CD1_1"]
    hdu[0].header["CD2_1"] = -hdu[0].header["CD2_1"]
    hdu[0].header["CRPIX1"] = int(hdu[0].header["NAXIS1"] / 2) + 1
    hdu[0].header["CRPIX2"] = int(hdu[0].header["NAXIS2"] / 2) + 1
    return hdu


def ls_open(ra, dec, size_arcsec, band="r", release="ls_dr9"):
    """
    Retrieves and opens a FITS cutout from the deepest stacked image in the
    specified Legacy Survey data release using the Astro Data Lab SIA service.

    Args:
        ra (float): Right Ascension in decimal degrees.
        dec (float): Declination in decimal degrees.
        size_arcsec (float): Size of the square cutout (side length) in arcseconds.
        band (str): The filter band (e.g., 'g', 'r', 'z'). Case-insensitive.
        release (str): The Legacy Survey Data Release (e.g., 'DR9').

    Returns:
        astropy.io.fits.HDUList: The opened FITS file object.
    """

    if sia is None:
        raise ImportError(
            "Cannot use ls_open without pyvo. Please install pyvo (pip install pyvo) before continuing."
        )

    # 1. Set the specific SIA service endpoint for the desired release
    # SIA endpoints for specific surveys are listed in the notebook.
    service_url = f"https://datalab.noirlab.edu/sia/{release.lower()}"
    svc = sia.SIAService(service_url)

    # 2. Convert size from arcseconds to degrees (FOV) for the SIA query
    # and apply the cosine correction for RA.
    fov_deg = size_arcsec / 3600.0

    # The search method takes the position (RA, Dec) and the square FOV.
    imgTable = svc.search(
        (ra, dec), (fov_deg / np.cos(dec * np.pi / 180.0), fov_deg), verbosity=2
    ).to_table()

    # 3. Filter the table for stacked images in the specified band
    target_band = band.lower()

    sel = (
        (imgTable["proctype"] == "Stack")
        & (imgTable["prodtype"] == "image")
        & (startswith(imgTable["obs_bandpass"].astype(str), target_band))
    )

    Table = imgTable[sel]

    if len(Table) == 0:
        raise ValueError(
            f"No stacked FITS image found for {release} band '{band}' at the requested RA {ra} and Dec {dec}."
        )

    # 4. Pick the "deepest" image (longest exposure time)
    # Note: 'exptime' data needs explicit float conversion for np.argmax
    max_exptime_index = np.argmax(Table["exptime"].data.data.astype("float"))
    row = Table[max_exptime_index]

    # 5. Download the file and open it
    url = row["access_url"]  # get the download URL

    # Use astropy's download_file, which handles the large data transfer
    # and automatically uses a long timeout (120s in the notebook example)
    filename = download_file(url, cache=False, show_progress=False, timeout=120)

    # Open the downloaded FITS file
    hdu = fits.open(filename)

    try:
        hdu = flip_hdu(hdu)
    except AssertionError:
        pass  # If WCS info is missing, skip flipping

    # Clean up the temporary file created by download_file
    try:
        os.remove(filename)
    except OSError:
        pass  # Ignore if cleanup fails

    return hdu
