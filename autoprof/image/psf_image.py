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
        self.resolutions[f"{self.pixelscale:.7e}"] = self
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

    @property
    def border(self):
        return self.pixelscale * (1 + np.array(self.data.shape)) / 2
    @property
    def border_int(self):
        return ((1 + np.array(self.data.shape)) / 2).astype(int)
    
    def get_resolution(self, resolution):
        if isinstance(resolution, float):
            res_str = f"{resolution:.7e}"
        elif isinstance(resolution, str):
            res_str = resolution
            
        if res_str in self.resolutions:
            return self.resolutions[res_str]

        if isinstance(resolution, str):
            res_flt = eval(resolution)
        else:
            res_flt = resolution
        resx = np.linspace(-0.5 + 1/res_flt, self.shape[1]/self.pixelscale - 0.5 - 1/res_flt, int(self.shape[1] / res_flt))
        resy = np.linspace(-0.5 + 1/res_flt, self.shape[0]/self.pixelscale - 0.5 - 1/res_flt, int(self.shape[0] / res_flt))
        new_psf = interpolate_Lanczos_grid(self.data, resx, resy, scale = 5)

        self.resolutions[res_str] = PSF_Image(
            new_psf,
            pixelscale = res_flt,
            zeropoint = self.zeropoint,
            rotation = self.rotation,
            note = self.note,
            origin = (-0.5 + 1/res_flt + self.origin[0], -0.5 + 1/res_flt + self.origin[1]),
            resolutions = {f"{self.pixelscale:.7e}": self},
        )
        return self.resolutions[res_str]
