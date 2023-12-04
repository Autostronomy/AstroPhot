from typing import Optional

import torch
import numpy as np

from ..param import Param_Unlock, Param_SoftLimits, Parameter_Node
from .model_object import Component_Model
from .core_model import AstroPhot_Model
from .psf_model_object import PSF_Model
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..image import PSF_Image, Window, Model_Image, Image
from ._shared_methods import select_target
from ..errors import SpecificationConflict

__all__ = ("Point_Source",)

class Point_Source(Component_Model):
    """Describes a point source in the image, this is a delta function at
    some position in the sky. This is typically used to describe
    stars, supernovae, very small galaxies, quasars, asteroids or any
    other object which can essentially be entirely described by a
    position and total flux (no structure).

    """
    model_type = f"point {Component_Model.model_type}"
    parameter_specs = {
        "flux": {"units": "log10(flux)"},
    }
    _parameter_order = Component_Model._parameter_order + ("flux",)
    useable = True


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.psf is None:
            raise ValueError("Point_Source needs psf information")

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        if parameters["flux"].value is not None:
            return
        target_area = target[self.window]
        target_dat = target_area.data.detach().cpu().numpy()
        with Param_Unlock(parameters["flux"]), Param_SoftLimits(parameters["flux"]):
            icenter = target_area.plane_to_pixel(parameters["center"].value)
            edge = np.concatenate(
                (
                    target_dat[:, 0],
                    target_dat[:, -1],
                    target_dat[0, :],
                    target_dat[-1, :],
                )
            )
            edge_average = np.median(edge)
            parameters["flux"].value = np.log10(np.abs(np.sum(target_dat - edge_average)))
            parameters["flux"].uncertainty = torch.std(target_area.data) / (np.log(10) * 10**parameters["flux"].value)

    # Psf convolution should be on by default since this is a delta function
    @property
    def psf_mode(self):
        return "full"
    @psf_mode.setter
    def psf_mode(self, value):
        pass
    
    def sample(
        self,
        image: Optional[Image] = None,
        window: Optional[Window] = None,
        parameters: Optional[Parameter_Node] = None,
    ):
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
        # Image on which to evaluate model
        if image is None:
            image = self.make_model_image(window=window)

        # Window within which to evaluate model
        if window is None:
            working_window = image.window.copy()
        else:
            working_window = window.copy()

        # Parameters with which to evaluate the model
        if parameters is None:
            parameters = self.parameters

        # Sample the PSF pixels
        if isinstance(self.psf, AstroPhot_Model):
            # Adjust for supersampled PSF
            psf_upscale = torch.round(self.psf.target.pixel_length / working_window.pixel_length).int()
            working_window = working_window.rescale_pixel(psf_upscale)
            working_window.shift(- parameters["center"].value)
            
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(
                window=working_window
            )

            # Fill the image using the PSF model
            psf = self.psf(
                image = working_image,
                parameters=parameters[self.psf.name],
            )

            # Scale for point source flux
            working_image.data *= 10**parameters["flux"].value

            # Return to original coordinates
            working_image.header.shift(parameters["center"].value)
            
        elif isinstance(self.psf, PSF_Image):
            psf = self.psf.copy()

            # Adjust for supersampled PSF
            psf_upscale = torch.round(psf.pixel_length / working_window.pixel_length).int()
            working_window = working_window.rescale_pixel(psf_upscale)
            
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(
                window=working_window
            )
            
            # Compute the center offset
            pixel_center = working_image.plane_to_pixel(parameters["center"].value)
            center_shift = pixel_center - torch.round(pixel_center)
            #working_image.header.pixel_shift(center_shift)
            psf.window.shift(working_image.pixel_to_plane(torch.round(pixel_center)))
            psf.data = self._shift_psf(psf = psf.data, shift = center_shift, shift_method = self.psf_subpixel_shift, keep_pad = False)
            psf.data /= torch.sum(psf.data)    
            
            # Scale for psf flux
            psf.data *= 10**parameters["flux"].value
            
            # Fill pixels with the PSF image
            working_image += psf
            
            # Shift image back to align with original pixel grid
            #working_image.header.pixel_shift(-center_shift)
            
        else:
            raise SpecificationConflict(f"Point_Source must have a psf that is either an AstroPhot_Model or a PSF_Image. not {type(self.psf)}")

        # Return to image pixelscale
        working_image = working_image.reduce(psf_upscale)
        if self.mask is not None:
            working_image.data = working_image.data * torch.logical_not(self.mask)

        # Add the sampled/integrated/convolved pixels to the requested image
        image += working_image

        return image        
