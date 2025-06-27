from typing import Optional

import torch
import numpy as np

from .model_object import ComponentModel
from ..utils.decorators import ignore_numpy_warnings
from ..image import Window, ModelImage
from ..errors import SpecificationConflict
from ..param import forward
from . import func

__all__ = ("PointSource",)


class PointSource(ComponentModel):
    """Describes a point source in the image, this is a delta function at
    some position in the sky. This is typically used to describe
    stars, supernovae, very small galaxies, quasars, asteroids or any
    other object which can essentially be entirely described by a
    position and total flux (no structure).

    """

    _model_type = "point"
    _parameter_specs = {
        "flux": {"units": "flux", "shape": ()},
    }
    _overload_parameter_specs = {
        "logflux": {
            "units": "log10(flux)",
            "shape": (),
            "overloads": "flux",
            "overload_function": lambda p: 10**p.logflux.value,
        }
    }
    usable = True

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.psf is None:
            raise SpecificationConflict("Point_Source needs psf information")

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if not hasattr(self, "logflux") or self.logflux.value is not None:
            return
        target_area = self.target[self.window]
        dat = target_area.data.npvalue.copy()
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.median(edge)
        self.logflux.dynamic_value = np.log10(np.abs(np.sum(dat - edge_average)))
        self.logflux.uncertainty = torch.std(dat) / np.sqrt(np.prod(dat.shape))

    # Psf convolution should be on by default since this is a delta function
    @property
    def psf_mode(self):
        return "full"

    @psf_mode.setter
    def psf_mode(self, value):
        pass

    @forward
    def sample(self, window: Optional[Window] = None, center=None, flux=None):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods and PSF
        convolution. This should not be overloaded except in special
        cases.

        This function is designed to compute the model on a given
        image or within a specified window. It takes care of sub-pixel
        sampling, recursive integration for high curvature regions,
        PSF convolution, and proper alignment of the computed model
        with the original pixel grid. The final model is then added to
        the requested image.

        Args:
          image (Optional[Image]): An AstroPhot Image object (likely a Model_Image)
                                     on which to evaluate the model values. If not
                                     provided, a new Model_Image object will be created.
          window (Optional[Window]): A window within which to evaluate the model.
                                   Should only be used if a subset of the full image
                                   is needed. If not provided, the entire image will
                                   be used.

        Returns:
          Image: The image with the computed model values.

        """
        # Window within which to evaluate model
        if window is None:
            window = self.window

        # Adjust for supersampled PSF
        psf_upscale = torch.round(self.target.pixel_length / self.psf.pixel_length).int().item()

        # Make the image object to which the samples will be tracked
        working_image = ModelImage(window=window, upsample=psf_upscale)

        # Compute the center offset
        pixel_center = torch.stack(working_image.plane_to_pixel(*center))
        pixel_shift = pixel_center - torch.round(pixel_center)
        psf = self.psf.data.value
        shift_kernel = func.fft_shift_kernel(psf.shape, pixel_shift[0], pixel_shift[1])
        psf = torch.fft.irfft2(shift_kernel * torch.fft.rfft2(psf, s=psf.shape), s=psf.shape)
        # (
        #     torch.nn.functional.conv2d(
        #         self.psf.data.value.view(1, 1, *self.psf.data.shape),
        #         shift_kernel.view(1, 1, *shift_kernel.shape),
        #         padding="valid",  # fixme add note about valid padding
        #     )
        #     .squeeze(0)
        #     .squeeze(0)
        # )
        psf = flux * psf

        # Fill pixels with the PSF image
        pixel_center = torch.round(pixel_center).int()
        psf_window = Window(
            (
                pixel_center[0] - psf.shape[0] // 2,
                pixel_center[0] + psf.shape[0] // 2 + 1,
                pixel_center[1] - psf.shape[1] // 2,
                pixel_center[1] + psf.shape[1] // 2 + 1,
            ),
            image=working_image,
        )
        working_image[psf_window].data._value += psf[working_image.get_other_indices(psf_window)]
        working_image = working_image.reduce(psf_upscale)

        # Return to image pixelscale
        if self.mask is not None:
            working_image.data = working_image.data.value * (~self.mask)

        return working_image
