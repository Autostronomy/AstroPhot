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
        fwhm = state.options["ap_gaussian_psf_fwhm", 2 * 2 * np.sqrt(2 * np.log(2))]
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # User specified psf size
        size = state.options["ap_gaussian_psf_size", 25]

        # User specified integration oversampling factor
        integrate_factor = state.options["ap_psf_integrate_factor", 5]

            
        # Create the cooridnate grid
        XX, YY = np.meshgrid(
            (np.arange(size*integrate_factor) - (size*integrate_factor - 1)/2) / integrate_factor,
            (np.arange(size*integrate_factor) - (size*integrate_factor - 1)/2) / integrate_factor,
        )

        # Compute radius to each pixel
        RR = np.sqrt(XX**2 + YY**2)

        # Evaluate the PSF function
        PSF = np.exp(- 0.5 * (RR / sigma)**2) / np.sqrt(2 * np.pi * sigma**2)
        if integrate_factor != 1:
            PSF = PSF.reshape(-1, integrate_factor, PSF.shape[0]//integrate_factor, integrate_factor).sum((-1,-3)) / (integrate_factor**2)
        # Add the PSF to the state
        state.data.update_psf(PSF, pixelscale = state.data.target.pixelscale, fwhm = fwhm)
        
        return state
