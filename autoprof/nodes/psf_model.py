from flow import Process
import numpy as np

class Gaussian_PSF(Process):
    """
    Construct a gaussian PSF for the image.
    """

    def action(self, state):

        # Do nothing if a PSF has already been given
        if state.data.psf is not None:
            return state
        
        # User specified fwhm
        if "ap_gaussian_psf_fwhm" in state.options:
            fwhm = state.options["ap_gaussian_psf_fwhm"]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            fwhm = 2 * 2 * np.sqrt(2 * np.log(2))
            sigma = 2.

        # User specified psf size
        if "ap_gaussian_psf_size" in state.options:
            size = state.options["ap_gaussian_psf_size"]
        else:
            size = 15

        # Create the cooridnate grid
        XX, YY = np.meshgrid(np.arange(size) - (size - 1)/2, np.arange(size) - (size - 1)/2)

        # Compute radius to each pixel
        RR = np.sqrt(XX**2 + YY**2)

        # Evaluate the PSF function
        PSF = np.exp(- 0.5 * (RR / sigma)**2) / np.sqrt(2 * np.pi * sigma**2)

        # Add the PSF to the state
        state.data.update_psf(PSF, pixelscale = state.data.target.pixelscale, fwhm = fwhm)

        return state
