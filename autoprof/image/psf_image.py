from .image_object import AP_Image
from autoprof.utils.interpolate import interpolate_Lanczos, interpolate_Lanczos_grid
from autoprof.utils.conversions.coordinates import coord_to_index, index_to_coord
import numpy as np

class PSF_Image(AP_Image):

    def __init__(self, data, origin = None, **kwargs):
        if origin is None:
            origin = - np.array(data.shape) * kwargs['pixelscale'] / 2
        super().__init__(data, origin = origin, **kwargs)

        self.data /= np.sum(self.data)
        self.resolutions = kwargs["resolutions"] if "resolutions" in kwargs else {}
        if "fwhm" in kwargs:
            self.fwhm = kwargs['fwhm']
        else:
            self.get_fwhm()

    def get_fwhm(self):

        center = coord_to_index(0., 0., self)
        central_flux = interpolate_Lanczos(self.data, X = center[1], Y = center[0], scale = 5)
        R = [0]
        flux = [central_flux]
        theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / 100), 100)
        while flux[-1] > (central_flux / 2) and R[-1] < (np.max(self.shape) / 2):
            R.append(R[-1] + 0.1)
            XX = R[-1] * np.cos(theta)
            YY = R[-1] * np.sin(theta)
            YY, XX = coord_to_index(XX, YY, self)
            flux.append(np.median(interpolate_Lanczos(self.data, XX, YY, scale = 5)))

        if R[-1] >= (np.max(self.shape) / 2):
            self.fwhm = (np.max(self.shape) / 2)
        else:
            self.fwhm = np.interp(central_flux / 2, list(reversed(flux)), list(reversed(R))) * 2
        
    def get_resolution(self, resolution):

        if str(resolution) in self.resolutions:
            return self.resolutions[str(resolution)]

        if isinstance(resolution, str):
            use_res = eval(resolution)
        else:
            use_res = resolution
        resx = np.linspace(-0.5 + 1/use_res, self.shape[1]/self.pixelscale - 0.5 - 1/use_res, int(self.shape[1]*use_res/self.pixelscale))
        resy = np.linspace(-0.5 + 1/use_res, self.shape[0]/self.pixelscale - 0.5 - 1/use_res, int(self.shape[0]*use_res/self.pixelscale))
        new_psf = interpolate_Lanczos_grid(self.data, resx, resy, scale = 5)

        self.resolutions[str(resolution)] = PSF_Image(
            new_psf,
            pixelscale = self.pixelscale / use_res,
            zeropoint = self.zeropoint,
            rotation = self.rotation,
            note = self.note,
            origin = (-0.5 + 1/use_res + self.origin[0], -0.5 + 1/use_res + self.origin[1]),
            resolutions = {str(1/use_res): self},
        )
        return self.resolutions[str(resolution)]
