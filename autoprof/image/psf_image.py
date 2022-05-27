from .image_object import AP_Image
from autoprof.utils.interpolate import interpolate_Lanczos
import numpy as np

class PSF_Image(AP_Image):

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        self.data /= np.sum(self.data)
        if "fwhm" in kwargs:
            self.fwhm = kwargs['fwhm']
        else:
            self.get_fwhm()

    def get_fwhm(self):

        center = ((self.shape[1] - 1) / 2, (self.shape[0] - 1) / 2)
        central_flux = interpolate_Lanczos(self.data, X = center[0], Y = center[1], scale = 5)
        R = [0]
        flux = [central_flux]
        theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / 100), 100)
        while flux[-1] > (central_flux / 2) and R[-1] < (np.max(self.data.shape) / 2):
            R.append(R[-1] + 0.1)
            XX = R[-1] * np.cos(theta) + center[0]
            YY = R[-1] * np.sin(theta) + center[1]
            flux.append(interpolate_Lanczos(self.data, XX, YY, scale = 5))

        if R[-1] >= (np.max(self.data.shape) / 2):
            self.fwhm = (np.max(self.data.shape) / 2)
        else:
            self.fwhm = np.interp(central_flux / 2, list(reversed(flux)), list(reversed(R))) * 2
        
