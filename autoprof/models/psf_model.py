from .star_model_object import Star_Model
from ..image import Model_Image
import torch
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from .. import AP_config

__all__ = ["PSF_Star"]

class PSF_Star(Star_Model):
    """Star model which uses an image of the PSF as it's representation
    for stars. Using Lanczos interpolation it will shift the PSF
    within a pixel to accurately represent the center location of a
    point source. There is no funcitonal form for this object type as
    any image can be supplied. Note that as an argument to the model
    at construction one can provide "psf" as an AutoProf Model_Image
    object. If the supplied image is at a higher resolution than the
    target image then the PSF will be upsampled at the time of
    sampling after the shift has been performed. In this way it is
    possible to get a more accurate representation of the PSF.

    Parameters:
        flux: the total flux of the star model, represented as the log of the total flux.
    """
    
    model_type = f"psf {Star_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = Star_Model._parameter_order + ("flux",)
    
    lanczos_kernel_size = 5
    clip_lanczos_kernel = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "psf" in kwargs:
            self.psf_model = kwargs["psf"]
        else:
            self.psf_model = Model_Image(data = torch.clone(self.target.psf), pixelscale = self.target.pixelscale)

    @torch.no_grad()
    def initialize(self, target = None):
        if target is None:
            target = self.target
        super().initialize(target)
        target_area = target[self.window]
        if self["flux"].value is None:
            self["flux"].set_value(torch.log10(torch.abs(torch.sum(target_area.data)) / target_area.pixelscale**2), override_locked = True)
        if self["flux"].uncertainty is None:
            self["flux"].set_uncertainty(torch.abs(self["flux"].value) * 1e-2, override_locked = True)
        
    def evaluate_model(self, image):
        new_origin = self["center"].value - self.psf_model.shape/2
        pixel_origin = torch.round(new_origin/image.pixelscale)*image.pixelscale
        pixel_shift = (new_origin/image.pixelscale - pixel_origin/image.pixelscale)*image.pixelscale
        LL = _shift_Lanczos_kernel_torch(-pixel_shift[0]/image.pixelscale, -pixel_shift[1]/image.pixelscale, 3, AP_config.ap_dtype, AP_config.ap_device)
        psf = Model_Image(
            data = torch.nn.functional.conv2d((torch.clone(self.psf_model.data)*((10**self["flux"].value)*image.pixelscale**2)).view(1,1,*self.psf_model.data.shape), LL.view(1,1,*LL.shape), padding = "same")[0][0],
            origin = pixel_origin - pixel_shift,
            pixelscale = self.psf_model.pixelscale,
        )
        img = image.blank_copy()
        img += psf
        return img.data
