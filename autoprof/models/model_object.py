from functools import partial
from typing import Optional, Union
import io

from torch.autograd.functional import jacobian
import numpy as np
import torch

from .core_model import AutoProf_Model
from ..image import Model_Image, Window
from .parameter_object import Parameter
from ..utils.initialize import center_of_mass
from ..utils.operations import fft_convolve_torch, fft_convolve_multi_torch
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from ..utils.conversions.coordinates import coord_to_index, index_to_coord
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["Component_Model"]


class Component_Model(AutoProf_Model):
    """Component_Model(name, target, window, locked, **kwargs)

    This is the basis for almost any model which represents a single
    object, or parametric form.  Subclassing models must define their
    parameters, initialization, and model evaluation
    functions. See individual models for their behaviour.

    """

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ("center",)

    # Technique and scope for PSF convolution
    psf_mode = "none"  # none, window/full
    # size in pixels of the PSF convolution box
    psf_window_size = 50
    # Integration scope for model
    integrate_mode = "window"  # none, window, full
    # size of the window in which to perform integration
    integrate_window_size = 10
    # Factor by which to upscale each dimension when integrating
    integrate_factor = 3  # number of pixels on one axis by which to supersample
    integrate_recursion_factor = 2  # relative size of windows between recursion levels (2 means each window will be half the size of the previous one)
    integrate_recursion_depth = (
        2  # number of recursion cycles to apply when integrating
    )
    jacobian_mode = "full"  # method to compute jacobian. "full" means full jacobian for all parameters at once (faster), "single" means one parameter at a time (less memory), "finite" means to use finite difference (minimum memory)

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    useable = False

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        # Set any user defined attributes for the model
        for kwarg in kwargs:  # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        self.parameter_specs = self.build_parameter_specs(
            kwargs.get("parameters", None)
        )
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.set_parameters(kwargs["parameters"])

        if "filename" in kwargs:
            self.load(kwargs["filename"])

    @property
    def parameters(self):
        try:
            return self._parameters
        except AttributeError:
            return {}

    @parameters.setter
    def parameters(self, val):
        self._parameters = val

    def parameter_order(self, parameters_identity: Optional[tuple] = None):
        """Returns a tuple of names of the parameters in their set order."""
        param_order = tuple()
        for P in self.__class__._parameter_order:
            if self[P].locked:
                continue
            if parameters_identity is not None and not any(
                pid in parameters_identity for pid in self[P].identities
            ):
                continue
            param_order = param_order + (P,)
        return param_order

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @select_target
    def initialize(self, target: Optional["Target_Image"] = None):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        super().initialize(target)
        # Get the sub-image area corresponding to the model image
        target_area = target[self.window]

        # Use center of window if a center hasn't been set yet
        if self["center"].value is None:
            self["center"].set_value(self.window.center, override_locked=True)
        else:
            return

        if self["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = coord_to_index(
            self["center"].value[0], self["center"].value[1], target_area
        )
        # Compute center of mass in window
        COM = center_of_mass(
            (
                init_icenter[0].detach().cpu().item(),
                init_icenter[1].detach().cpu().item(),
            ),
            target_area.data.detach().cpu().numpy(),
        )
        if np.any(np.array(COM) < 0) or np.any(
            np.array(COM) >= np.array(target_area.data.shape)
        ):
            AP_config.ap_logger.warning("center of mass failed, using center of window")
            return
        # Convert center of mass indices to coordinates
        COM_center = index_to_coord(COM[0], COM[1], target_area)
        # Set the new coordinates as the model center
        self["center"].value = COM_center

    # Fit loop functions
    ######################################################################
    def evaluate_model(self, image: "BaseImage"):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        Args:
          image (BaseImage): The image defining the set of pixels on which to evaluate the model

        """
        return torch.zeros_like(image.data)  # do nothing in base model

    def sample(
        self,
        image: Optional["BaseImage"] = None,
        window: Optional[Window] = None,
    ):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods and PSF
        convolution. This should not be overloaded except in special
        cases.

        Parameters:
            image (Optional[BaseImage]): An AutoProf Image object (likely a Model_Image) on which to evaluate the model values.
            window (Optional[Window]): A window within which to evaluate the model. Should only be used if a subset of the full image is needed.

        """
        # Image on which to evaluate model
        if image is None:
            image = self.make_model_image(window=window)

        # Window within which to evaluate model
        if window is None:
            working_window = image.window.make_copy()
        else:
            working_window = window.make_copy() & image.window

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        if "full" in self.psf_mode:
            # Add border for psf convolution edge effects, will be cropped out later
            working_window += self.target.psf_border
            # Determine the pixels scale at which to evalaute, this is smaller if the PSF is upscaled
            working_pixelscale = image.pixelscale / self.target.psf_upscale
            # Sub pixel shift to align the model with the center of a pixel
            align = self.target.pixel_center_alignment()
            center_shift = (
                self["center"].value
                - (
                    torch.round(self["center"].value / working_pixelscale - align)
                    + align
                )
                * working_pixelscale
            )
            working_window.shift_origin(center_shift)
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(
                pixelscale=working_pixelscale, window=working_window
            )
            # Evaluate the model at the current resolution
            working_image.data += self.evaluate_model(working_image)
            # If needed, super-resolve the image in areas of high curvature so pixels are properly sampled
            self.integrate_model(
                working_image,
                self.integrate_window(working_image, "center"),
                self.integrate_recursion_depth,
            )
            # Convolve the PSF
            LL = _shift_Lanczos_kernel_torch(
                -center_shift[0] / working_image.pixelscale,
                -center_shift[1] / working_image.pixelscale,
                3,
                AP_config.ap_dtype,
                AP_config.ap_device,
            )
            shift_psf = torch.nn.functional.conv2d(
                self.target.psf.view(1, 1, *self.target.psf.shape),
                LL.view(1, 1, *LL.shape),
                padding="same",
            )[0][0]
            working_image.data = fft_convolve_torch(
                working_image.data, shift_psf / torch.sum(shift_psf), img_prepadded=True
            )
            # Shift image back to align with original pixel grid
            working_image.window.shift_origin(-center_shift)
            # Add the sampled/integrated/convolved pixels to the requested image
            image += working_image.reduce(self.target.psf_upscale).crop(
                self.target.psf_border_int
            )
        else:
            # Create an image to store pixel samples
            working_image = Model_Image(
                pixelscale=image.pixelscale, window=working_window
            )
            # Evaluate the model on the image
            working_image.data += self.evaluate_model(working_image)
            # Super-resolve and integrate where needed
            self.integrate_model(
                working_image,
                self.integrate_window(working_image, "pixel"),
                self.integrate_recursion_depth,
            )
            # Add the sampled/integrated pixels to the requested image
            image += working_image

        return image

    def integrate_model(
        self, working_image: "BaseImage", window: Window, depth: int = 2
    ):
        """Sample the model at a higher resolution than the given image, then
        integrate the super resolution up to the image
        resolution.

        Parameters:
            working_image (BaseImage): the image on which to perform the model integration. Pixels in this image will be replaced with the integrated values
            window (Window): A Window object within which to perform the integration.
            depth (int): recursion depth tracker. When called with depth = n, this function will call itself again with depth = n-1 until depth is 0 at which point it will exit without integrating.

        """
        if depth <= 0 or "none" in self.integrate_mode:
            return
        # Determine the on-sky window in which to integrate
        try:
            if window.overlap_frac(working_image.window) <= 0.0:
                return
        except AssertionError:
            return
        # Only need to evaluate integration within working image
        working_window = window & working_image.window
        # Determine the upsampled pixelscale
        integrate_pixelscale = working_image.pixelscale / self.integrate_factor
        # Build an image to hold the integration data
        integrate_image = Model_Image(
            pixelscale=integrate_pixelscale, window=working_window
        )
        # Evaluate the model at the fine sampling points
        integrate_image.data = self.evaluate_model(integrate_image)

        # If needed, recursively evaluates smaller windows
        recursive_shape = (
            window.shape / integrate_pixelscale
        )  # get the number of pixels across the integrate window
        recursive_shape = torch.round(
            recursive_shape / self.integrate_recursion_factor
        ).int()  # divide window by recursion factor, ensure integer result
        window_align = torch.isclose(
            (
                ((window.center - integrate_image.origin) / integrate_image.pixelscale)
                % 1
            ),
            torch.tensor(0.5, dtype=AP_config.ap_dtype, device=AP_config.ap_device),
            atol=0.25,
        )
        recursive_shape = (
            recursive_shape
            + 1
            - (recursive_shape % 2)
            + 1
            - window_align.to(dtype=torch.int32)
        ) * integrate_pixelscale  # ensure shape pairity is matched during recursion
        self.integrate_model(
            integrate_image,
            Window(
                center=window.center,
                shape=recursive_shape,
            ),
            depth=depth - 1,
        )
        # Replace the image data where the integration has been done
        working_image.replace(integrate_image.reduce(self.integrate_factor))

    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
        """Compute the jacobian for this model. Done by first constructing a
        full jacobian (Npixels * Nparameters) of zeros then call the
        jacobian method of each sub model and add it in to the total.

        Args:
          parameters (Optional[torch.Tensor]): 1D parameter vector to overwrite current values
          as_representation (bool): Indiates if the "parameters" argument is in the form of the real values, or as representations in the (-inf,inf) range. Default False
          pass_jacobian (Optional["Jacobian_Image"]): A Jacobian image pre-constructed to be passed along instead of constructing new Jacobians

        """
        if window is None:
            window = self.window
        else:
            window = self.window & window
        # skip jacobian calculation if no parameters match criteria
        porder = self.parameter_order(parameters_identity=parameters_identity)
        if len(porder) == 0 or window.overlap_frac(self.window) <= 0:
            return self.target[window].jacobian_image()

        if parameters is not None:
            self.set_parameters(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )

        if parameters_identity is None:
            pids = None
        else:
            pids = self.get_parameter_identity_vector(
                parameters_identity=parameters_identity,
            )
        if self.jacobian_mode == "full":
            full_jac = jacobian(
                lambda P: self(
                    image=None,
                    parameters=P,
                    as_representation=as_representation,
                    parameters_identity=pids,
                    window=window,
                ).data,
                self.get_parameter_vector(
                    as_representation=as_representation,
                    parameters_identity=parameters_identity,
                ).detach(),
                strategy="forward-mode",
                vectorize=True,
                create_graph=False,
            )
        elif self.jacobian_mode == "single":
            raise NotImplementedError("single jacobian not avaialble yet")
        elif self.jacobian_mode == "chunk":
            raise NotImplementedError("chunk jacobian not avaialble yet")
        else:
            raise ValueError(
                f"Unrecognized jacobian mode for {self.name}: {self.jacobian_mode}"
            )

        jac_img = self.target[window].jacobian_image(
            parameters=self.get_parameter_identity_vector(
                parameters_identity=parameters_identity,
            ),
            data=full_jac,
        )
        return jac_img

    def get_state(self):
        """Returns a dictionary with a record of the current state of the
        model.

        Specifically, the current parameter settings and the
        window for this model. From this information it is possible
        for the model to re-build itself lated when loading from
        disk. Note that the target image is not saved, this must be
        reset when loading the model.

        """
        state = super().get_state()
        state["window"] = self.window.get_state()
        if "parameters" not in state:
            state["parameters"] = {}
        for P in self.parameters:
            state["parameters"][P] = self[P].get_state()
        return state

    def load(self, filename: Union[str, dict, io.TextIOBase] = "AutoProf.yaml"):
        """Used to load the model from a saved state.

        Sets the model window to the saved value and updates all
        parameters with the saved information. This overrides the
        current parameter settings.

        Args:
          filename: The source from which to load the model parameters. Can be a string (the name of the file on disc), a dictionary (formatted as if from self.get_state), or an io.TextIOBase (a file stream to load the file from).

        """
        state = AutoProf_Model.load(filename)
        self.name = state["name"]
        self.window = Window(**state["window"])
        for key in state["parameters"]:
            self[key].update_state(state["parameters"][key])
            self[key].to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        return state

    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import integrate_window
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
    from ._model_methods import __getitem__
    from ._model_methods import __contains__
    from ._model_methods import __str__
