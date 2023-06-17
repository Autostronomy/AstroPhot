from typing import Optional

import torch

from ..utils.decorators import default_internal
from ..image import PSF_Image
from .auxiliary_model import Auxiliary_Model
from ..utils.parametric_profiles import (
    moffat_torch,
)

__all__ = ("PSF_Aux", "Circular_PSF_Aux", "Moffat_Circular_PSF_Aux")

class PSF_Aux(Auxiliary_Model):

    model_type = f"psf {Auxiliary_Model.model_type}"
    useable = False
    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-5
    # Method for initial sampling of model
    sampling_mode = "midpoint" # midpoint, trapezoid, simpson
    
    # Level to which each pixel should be evaluated
    sampling_tolerance = 1e-2
    
    # Integration scope for model
    integrate_mode = "threshold"  # none, threshold, full*
    
    def sample(
        self,
        image: Optional["Image"] = None,
        parameters: Optional["Parameter_Group"] = None,
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
          image (Optional[Image]): An AutoPhot Image object (likely a Model_Image)
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
            image = self.target.psf.blank_copy()
            
        # Parameters with which to evaluate the model
        if parameters is None:
            parameters = self.parameters

        # Create an image to store pixel samples
        working_image = PSF_Image(
            data = torch.zeros_like(image.data), pixelscale=image.pixelscale, center = image.center,
        )
        # Evaluate the model on the image
        reference, deep = self._sample_init(
            image=working_image, parameters=parameters, center = image.center,
        )
        # Super-resolve and integrate where needed
        deep = self._sample_integrate(deep, reference, working_image, parameters, center = image.center)
        # Add the sampled/integrated pixels to the requested image
        working_image.data += deep
        
        image += working_image

        return image
            
    # Extra background methods
    ######################################################################
    from ._model_methods import _sample_init
    from ._model_methods import _sample_integrate
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters

class Circular_PSF_Aux(PSF_Aux):

    model_type = f"circular {PSF_Aux.model_type}"
    useable = False

    @default_internal
    def radius_metric(self, X, Y, image=None, parameters=None):
        return torch.sqrt(
            (torch.abs(X)) ** 2 + (torch.abs(Y)) ** 2
        )
    
    @default_internal
    def evaluate_model(self, X = None, Y = None, image = None, parameters = None):
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - image.center[...,None, None]
        return self.radial_model(
            self.radius_metric(X, Y, image, parameters),
            image,
            parameters,
            
        )
        
    
class Moffat_Circular_PSF_Aux(Circular_PSF_Aux):

    model_type = f"moffat {Circular_PSF_Aux.model_type}"
    parameter_specs = {
        "n*": {"units": "none", "limits": (0.1, 10), "uncertainty": 0.05},
        "Rd*": {"units": "arcsec", "limits": (0, None)},
        "I0*": {"value": 1., "locked": True, "units": "log10(flux/arcsec^2)"},
    }
    _parameter_order = PSF_Aux._parameter_order + ("n*", "Rd*", "I0*")
    useable = True
    
    @default_internal
    def radial_model(self, R, image=None, parameters=None):
        return moffat_torch(
            R + self.softening,
            parameters["n*"].value,
            parameters["Rd*"].value,
            (10 ** parameters["I0*"].value) * image.pixel_area,
        )
    
