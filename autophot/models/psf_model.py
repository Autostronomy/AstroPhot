import torch

from .star_model_object import Star_Model
from ..image import Model_Image
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["PSF_Star"]


class PSF_Star(Star_Model):
    """Star model which uses an image of the PSF as it's representation
    for stars. Using Lanczos interpolation it will shift the PSF
    within a pixel to accurately represent the center location of a
    point source. There is no funcitonal form for this object type as
    any image can be supplied. Note that as an argument to the model
    at construction one can provide "psf" as an AutoPhot Model_Image
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
    useable = True

    lanczos_kernel_size = 5
    clip_lanczos_kernel = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fixme, model already has PSF interface, those can just be merged
        if "psf" in kwargs:
            self.psf_model = kwargs["psf"]
        else:
            self.psf_model = Model_Image(
                data=torch.clone(self.psf.data),
                pixelscale=self.psf.pixelscale,
            )

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)
        target_area = target[self.window]
        if parameters["flux"].value is None:
            parameters["flux"].set_value(
                torch.log10(
                    torch.abs(torch.sum(target_area.data)) / target_area.pixel_area
                ),
                override_locked=True,
            )
        if parameters["flux"].uncertainty is None:
            parameters["flux"].set_uncertainty(
                torch.abs(parameters["flux"].value) * 1e-2, override_locked=True
            )

    @default_internal
    def evaluate_model(self, X=None, Y=None, image=None, parameters=None, **kwargs):
        new_origin = parameters["center"].value - self.psf_model.window.end / 2
        pixel_origin = image.pixel_to_world(torch.round(image.world_to_pixel(new_origin)))
        pixel_shift = (
            image.world_to_pixel(new_origin) - image.world_to_pixel(pixel_origin)
        )
        LL = _shift_Lanczos_kernel_torch(
            -pixel_shift[0],
            -pixel_shift[1],
            3,
            AP_config.ap_dtype,
            AP_config.ap_device,
        )
        psf = Model_Image(
            data=torch.nn.functional.conv2d(
                (
                    torch.clone(self.psf_model.data)
                    * ((10 ** parameters["flux"].value) * image.pixel_area)
                ).view(1, 1, *self.psf_model.data.shape),
                LL.view(1, 1, *LL.shape),
                padding="same",
            )[0][0],
            origin=new_origin,
            pixelscale=self.psf_model.pixelscale,
        )

        # fixme pick nearest neighbor for each X, Y? interpolate?
        img = image.blank_copy()
        img += psf
        return img.data
