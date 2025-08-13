from typing import Optional

import torch
import numpy as np

from .base import Model
from .model_object import ComponentModel
from ..image import ModelImage
from ..utils.decorators import ignore_numpy_warnings, combine_docstrings
from ..utils.interpolate import interp2d
from ..image import Window, PSFImage
from ..errors import SpecificationConflict
from ..param import forward
from ..backend_obj import backend, ArrayLike

__all__ = ("PointSource",)


@combine_docstrings
class PointSource(ComponentModel):
    """Describes a point source in the image, this is a delta function at
    some position in the sky. This is typically used to describe
    stars, supernovae, very small galaxies, quasars, asteroids or any
    other object which can essentially be entirely described by a
    position and total flux (no structure).

    **Parameters:**
    -    `flux`: The total flux of the point source

    """

    _model_type = "point"
    _parameter_specs = {
        "flux": {"units": "flux", "valid": (0, None), "shape": ()},
    }
    usable = True

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.psf is None:
            raise SpecificationConflict("Point_Source needs a psf!")

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        if self.flux.initialized:
            return
        target_area = self.target[self.window]
        dat = backend.to_numpy(target_area.data).copy()
        edge = np.concatenate((dat[:, 0], dat[:, -1], dat[0, :], dat[-1, :]))
        edge_average = np.median(edge)
        self.flux.dynamic_value = np.abs(np.sum(dat - edge_average))

    # Psf convolution should be on by default since this is a delta function
    @property
    def psf_convolve(self):
        return True

    @psf_convolve.setter
    def psf_convolve(self, value):
        pass

    @property
    def integrate_mode(self):
        return "none"

    @integrate_mode.setter
    def integrate_mode(self, value):
        pass

    @forward
    def sample(
        self,
        window: Optional[Window] = None,
        center: ArrayLike = None,
        flux: ArrayLike = None,
    ) -> ModelImage:
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

        if isinstance(self.psf, PSFImage):
            psf = self.psf.data
        elif isinstance(self.psf, Model):
            psf = self.psf().data
        else:
            raise TypeError(
                f"PSF must be a PSFImage or Model instance, got {type(self.psf)} instead."
            )

        # Make the image object to which the samples will be tracked
        working_image = self.target[window].model_image(upsample=self.psf_upscale)

        i, j = working_image.pixel_center_meshgrid()
        i0, j0 = working_image.plane_to_pixel(*center)
        working_image.data = interp2d(
            psf, i - i0 + (psf.shape[0] // 2), j - j0 + (psf.shape[1] // 2)
        )

        working_image.data = flux * working_image.data

        working_image = working_image.reduce(self.psf_upscale)

        return working_image
