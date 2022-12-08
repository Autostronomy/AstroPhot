from .star_model_object import Star_Model
from autoprof.image import Model_Image
import torch

__all__ = ["PSF_Star"]

class PSF_Star(Star_Model):
    """Star model which uses an image of the PSF as it's representation
    for stars. Using Lanczos interpolation it will shift the PSF
    within a pixel to accurately represent the center location of a
    point source.

    """
    
    model_type = f"psf {Star_Model.model_type}"
    parameter_specs = {
        "sky": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("sky",)
    
    lanczos_kernel_size = 5
    clip_lanczos_kernel = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf_model = Model_Image(data = torch.clone(self.target.psf), pixelscale = self.target.pixelscale)

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        target_area = target[self.fit_window]        
        self["sky"].set_value(torch.log10(torch.abs(torch.sum(target_area.data)) / target_area.pixelscale**2), override_locked = self["sky"].value is None)
        self["sky"].set_uncertainty(torch.abs(self["sky"].value) * 1e-2, override_locked = self["sky"].uncertainty is None)
        
    def evaluate_model(self, image):# fixme this definitely has bugs
        new_origin = self["center"].value - self.psf_model.shape/2
        pixel_origin = torch.round(new_origin/image.pixelscale)*image.pixelscale
        pixel_shift = (new_origin/image.pixelscale - pixel_origin/image.pixelscale)*image.pixelscale
        psf = Model_Image(data = torch.clone(self.psf_model.data)*((10**self["sky"].value)*image.pixelscale**2), origin = pixel_origin - pixel_shift, pixelscale = self.psf_model.pixelscale)
        psf.shift_origin(pixel_shift)
        img = image.blank_copy()
        img += psf
        return img.data
