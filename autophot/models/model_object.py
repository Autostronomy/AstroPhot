from functools import partial
from typing import Optional, Union
import io

from torch.autograd.functional import jacobian
import numpy as np
import torch
import matplotlib.pyplot as plt

from .core_model import AutoPhot_Model
from ..image import Model_Image, Window, PSF_Image, Jacobian_Image, Window_List
from .parameter_object import Parameter
from .parameter_group import Parameter_Group
from ..utils.initialize import center_of_mass
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.operations import (
    fft_convolve_torch,
    fft_convolve_multi_torch,
    selective_integrate,
)
from ..utils.interpolate import _shift_Lanczos_kernel_torch
from ._shared_methods import select_target
from .. import AP_config

__all__ = ["Component_Model"]


class Component_Model(AutoPhot_Model):
    """Component_Model(name, target, window, locked, **kwargs)

    Component_Model is a base class for models that represent single
    objects or parametric forms. It provides the basis for subclassing
    models and requires the definition of parameters, initialization,
    and model evaluation functions. This class also handles
    integration, PSF convolution, and computing the Jacobian matrix.

    Attributes:
      parameter_specs (dict): Specifications for the model parameters.
      _parameter_order (tuple): Fixed order of parameters.
      psf_mode (str): Technique and scope for PSF convolution.
      psf_window_size (int): Size in pixels of the PSF convolution box.
      integrate_mode (str): Integration scope for the model.
      integrate_window_size (int): Size of the window in which to perform integration.
      integrate_factor (int): Factor by which to upscale each dimension when integrating.
      integrate_recursion_factor (int): Relative size of windows between recursion levels.
      integrate_recursion_depth (int): Number of recursion cycles to apply when integrating.
      jacobian_chunksize (int): Maximum size of parameter list before jacobian will be broken into smaller chunks.
      special_kwargs (list): Parameters which are treated specially by the model object and should not be updated directly.
      useable (bool): Indicates if the model is useable.

    Methods:
      initialize: Determine initial values for the center coordinates.
      sample: Evaluate the model on the space covered by an image object.
      jacobian: Compute the Jacobian matrix for this model.

    """

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {
        "center": {"units": "arcsec", "uncertainty": 0.1},
    }
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ("center",)

    # Scope for PSF convolution
    psf_mode = "none"  # none, full
    # Technique for PSF convolution
    psf_convolve_mode = "fft" # fft, direct
    # method for initial sampling of grid before subpixel integration
    sampling_mode = "midpoint" # midpoint, trapezoid, simpson 
    # Integration scope for model
    integrate_mode = "threshold"  # none, threshold, full*

    # Number of recursion cycles to apply when integrating (threshold mode)
    integrate_recursion_depth = 3
    # Threshold for triggering pixel integration (threshold mode)
    integrate_threshold = 1e-2

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_chunksize = 10

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    track_attrs = [
        "psf_mode",
        "psf_convolve_mode",
        "sampling_mode",
        "integrate_mode",
        "integrate_recursion_depth",
        "integrate_threshold",
        "jacobian_chunksize",
    ]
    useable = False

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.psf = None

        # Set any user defined attributes for the model
        for kwarg in kwargs:  # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        # If loading from a file, get model configuration then exit __init__
        if "filename" in kwargs:
            self.load(kwargs["filename"])
            return

        self.parameter_specs = self.build_parameter_specs(
            kwargs.get("parameters", None)
        )
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.parameters.set_values(kwargs["parameters"])

    @property
    def psf(self):
        if self._psf is None:
            return self.target.psf
        return self._psf

    @psf.setter
    def psf(self, val):
        if val is None:
            self._psf = None
        elif isinstance(val, PSF_Image):
            self._psf = val
        else:
            self._psf = PSF_Image(
                val,
                pixelscale=self.target.pixelscale,
                band=self.target.band,
            )

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self,
        target: Optional["Target_Image"] = None,
        parameters: Optional[Parameter_Group] = None,
        **kwargs,
    ):
        """Determine initial values for the center coordinates. This is done
        with a local center of mass search which iterates by finding
        the center of light in a window, then iteratively updates
        until the iterations move by less than a pixel.

        Args:
          target (Optional[Target_Image]): A target image object to use as a reference when setting parameter values

        """
        super().initialize(target=target, parameters=parameters)
        # Get the sub-image area corresponding to the model image
        target_area = target[self.window]

        # Use center of window if a center hasn't been set yet
        if parameters["center"].value is None:
            parameters["center"].set_value(self.window.center, override_locked=True)
        else:
            return

        if parameters["center"].locked:
            return

        # Convert center coordinates to target area array indices
        init_icenter = target_area.world_to_pixel(parameters["center"].value)
        
        # Compute center of mass in window
        COM = center_of_mass(
            (
                init_icenter[1].detach().cpu().item(),
                init_icenter[0].detach().cpu().item(),
            ),
            target_area.data.detach().cpu().numpy(),
        )
        if np.any(np.array(COM) < 0) or np.any(
            np.array(COM) >= np.array(target_area.data.shape)
        ):
            AP_config.ap_logger.warning("center of mass failed, using center of window")
            return
        COM = (COM[1], COM[0])
        # Convert center of mass indices to coordinates
        COM_center = target_area.pixel_to_world(torch.tensor(COM, dtype=AP_config.ap_dtype, device=AP_config.ap_device))

        # Set the new coordinates as the model center
        parameters["center"].value = COM_center

    # Fit loop functions
    ######################################################################
    def evaluate_model(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        image: Optional["Image"] = None,
        parameters: "Parameter_Group" = None,
        **kwargs,
    ):
        """Evaluate the model on every pixel in the given image. The
        basemodel object simply returns zeros, this function should be
        overloaded by subclasses.

        Args:
          image (Image): The image defining the set of pixels on which to evaluate the model

        """
        if X is None or Y is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[...,None, None]
        return torch.zeros_like(X)  # do nothing in base model

    def sample(
        self,
        image: Optional["Image"] = None,
        window: Optional[Window] = None,
        parameters: Optional[Parameter_Group] = None,
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
            image = self.make_model_image(window=window)

        # Window within which to evaluate model
        if window is None:
            working_window = image.window.copy()
        else:
            working_window = window.copy()

        # Parameters with which to evaluate the model
        if parameters is None:
            parameters = self.parameters

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        if "full" in self.psf_mode:
            # Add border for psf convolution edge effects, will be cropped out later
            working_window += self.psf.psf_border
            # Determine the pixels scale at which to evalaute, this is smaller if the PSF is upscaled
            working_pixelscale = image.pixelscale / self.psf.psf_upscale
            # Make the image object to which the samples will be tracked
            working_image = Model_Image(
                pixelscale=working_pixelscale, window=working_window
            )            
            # Sub pixel shift to align the model with the center of a pixel
            pixel_center = working_image.world_to_pixel(parameters["center"].value)
            center_shift = pixel_center - torch.round(pixel_center)
            working_image.header.pixel_shift_origin(center_shift)
            # Evaluate the model at the current resolution
            working_image.data += self.evaluate_model(
                image=working_image, parameters=parameters
            )
            # If needed, super-resolve the image in areas of high curvature so pixels are properly sampled
            if self.integrate_mode == "none":
                pass
            elif self.integrate_mode == "threshold":
                Coords = working_image.get_coordinate_meshgrid()
                X, Y = Coords - parameters["center"].value[...,None, None]
                selective_integrate(
                    X=X,
                    Y=Y,
                    data=working_image.data,
                    image_header=working_image.header,
                    eval_brightness=self.evaluate_model,
                    eval_parameters=parameters,
                    max_depth=self.integrate_recursion_depth,
                    integrate_threshold=self.integrate_threshold,
                )
            else:
                raise ValueError(
                    f"{self.name} has unknown integration mode: {self.integrate_mode}"
                )
            # Convolve the PSF
            pix_center_shift = working_image.world_to_pixel_delta(center_shift)
            LL = _shift_Lanczos_kernel_torch(
                -pix_center_shift[0],
                -pix_center_shift[1],
                3,
                AP_config.ap_dtype,
                AP_config.ap_device,
            )
            shift_psf = torch.nn.functional.conv2d(
                self.psf.data.view(1, 1, *self.psf.data.shape),
                LL.view(1, 1, *LL.shape),
                padding="same",
            ).squeeze()
            # Remove unphysical negative pixels from Lanczos interpolation
            shift_psf[shift_psf < 0] = torch.tensor(0., dtype=AP_config.ap_dtype, device = AP_config.ap_device)
            if self.psf_convolve_mode == "fft":
                working_image.data = fft_convolve_torch(
                    working_image.data, shift_psf / torch.sum(shift_psf), img_prepadded=True
                )
            elif self.psf_convolve_mode == "direct":
                working_image.data = torch.nn.functional.conv2d(
                    working_image.data.view(1, 1, *working_image.data.shape),
                    torch.flip(shift_psf.view(1, 1, *shift_psf.shape) / torch.sum(shift_psf), dims = (2,3)),
                    padding="same",
                ).squeeze()
            else:
                raise ValueError(f"unrecognized psf_convolve_mode: {self.psf_convolve_mode}")
                
            # Shift image back to align with original pixel grid
            working_image.header.shift_origin(-center_shift)
            # Add the sampled/integrated/convolved pixels to the requested image
            working_image = working_image.reduce(self.psf.psf_upscale).crop(
                self.psf.psf_border_int
            )
            if self.mask is not None:
                working_image.data = working_image.data * torch.logical_not(self.mask)
            image += working_image

        else:
            # Create an image to store pixel samples
            working_image = Model_Image(
                pixelscale=image.pixelscale, window=working_window
            )
            # Evaluate the model on the image
            working_image.data += self.evaluate_model(
                image=working_image, parameters=parameters
            )
            # Super-resolve and integrate where needed
            if self.integrate_mode == "none":
                pass
            elif self.integrate_mode == "threshold":
                Coords = working_image.get_coordinate_meshgrid()
                X, Y = Coords - parameters["center"].value[...,None, None]
                selective_integrate(
                    X=X,
                    Y=Y,
                    data=working_image.data,
                    image_header=working_image.header,
                    eval_brightness=self.evaluate_model,
                    eval_parameters=parameters,
                    max_depth=self.integrate_recursion_depth,
                    integrate_threshold=self.integrate_threshold,
                )
            else:
                raise ValueError(
                    f"{self.name} has unknown integration mode: {self.integrate_mode}"
                )
            # Add the sampled/integrated pixels to the requested image
            if self.mask is not None:
                working_image.data = working_image.data * torch.logical_not(self.mask)
            image += working_image

        return image

    @torch.no_grad()
    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        window: Optional[Window] = None,
        pass_jacobian: Optional[Jacobian_Image] = None,
        **kwargs,
    ):
        """Compute the Jacobian matrix for this model.

        The Jacobian matrix represents the partial derivatives of the
        model's output with respect to its input parameters. It is
        useful in optimization and model fitting processes. This
        method simplifies the process of computing the Jacobian matrix
        for astronomical image models and is primarily used by the
        Levenberg-Marquardt algorithm for model fitting tasks.

        Args:
          parameters (Optional[torch.Tensor]): A 1D parameter tensor to override the
                                               current model's parameters.
          as_representation (bool): Indicates if the parameters argument is
                                    provided as real values or representations
                                    in the (-inf, inf) range. Default is False.
          parameters_identity (Optional[tuple]): Specifies which parameters are to be
                                                 considered in the computation.
          window (Optional[Window]): A window object specifying the region of interest
                                     in the image.
          **kwargs: Additional keyword arguments.

        Returns:
          Jacobian_Image: A Jacobian_Image object containing the computed Jacobian matrix.

        """
        if window is None:
            window = self.window
        else:
            if isinstance(window, Window_List):
                window = window.window_list[pass_jacobian.index(self.target)]
            window = self.window & window

        # skip jacobian calculation if no parameters match criteria
        porder = self.parameters.order(parameters_identity=parameters_identity)
        if len(porder) == 0 or window.overlap_frac(self.window) <= 0:
            return self.target[window].jacobian_image()

        # Set the parameters if provided and check the size of the parameter list
        dochunk = False
        if parameters is not None:
            if len(parameters) > self.jacobian_chunksize:
                dochunk = True
            self.parameters.set_values(
                parameters,
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            )
        else:
            if (
                len(
                    self.parameters.get_identity_vector(
                        parameters_identity=parameters_identity
                    )
                )
                > self.jacobian_chunksize
            ):
                dochunk = True

        # If the parameter list is too large, apply the chunk jacobian analysis
        if dochunk:
            return self._chunk_jacobian(
                as_representation=as_representation,
                parameters_identity=parameters_identity,
                window=window,
                **kwargs,
            )

        # Store the parameter identities
        if parameters_identity is None:
            pids = None
        else:
            pids = self.parameters.get_identity_vector(
                parameters_identity=parameters_identity,
            )
        # Compute the jacobian
        full_jac = jacobian(
            lambda P: self(
                image=None,
                parameters=P,
                as_representation=as_representation,
                parameters_identity=pids,
                window=window,
            ).data,
            self.parameters.get_vector(
                as_representation=as_representation,
                parameters_identity=parameters_identity,
            ).detach(),
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

        # Store the jacobian as a Jacobian_Image object
        jac_img = self.target[window].jacobian_image(
            parameters=self.parameters.get_identity_vector(
                parameters_identity=parameters_identity,
            ),
            data=full_jac,
        )
        return jac_img

    @torch.no_grad()
    def _chunk_jacobian(
        self,
        as_representation: bool = False,
        parameters_identity: Optional[tuple] = None,
        window: Optional[Window] = None,
        **kwargs,
    ):
        """Evaluates the Jacobian in small chunks to reduce memory usage.

        For models with many parameters it can be prohibitive to build
        the full Jacobian in a single pass. Instead this function
        breaks the list of parameters into chunks as determined by
        `self.jacobian_chunksize` evaluates the Jacobian only for
        those, it then builds up the full Jacobian as a separate
        tensor. This is for internal use and should be called by the
        `self.jacobian` function when appropriate.

        """

        pids = self.parameters.get_identity_vector(
            parameters_identity=parameters_identity,
        )
        jac_img = self.target[window].jacobian_image(
            parameters=pids,
        )

        for ichunk in range(0, len(pids), self.jacobian_chunksize):
            jac_img += self.jacobian(
                parameters=None,
                as_representation=as_representation,
                parameters_identity=pids[ichunk : ichunk + self.jacobian_chunksize],
                window=window,
                **kwargs,
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
        state["parameter_order"] = list(self.parameter_order)
        if "parameters" not in state:
            state["parameters"] = {}
        for P in self.parameters:
            state["parameters"][P.name] = P.get_state()
        for key in self.track_attrs:
            if getattr(self, key) != getattr(self.__class__, key):
                state[key] = getattr(self, key)
        return state

    def load(self, filename: Union[str, dict, io.TextIOBase] = "AutoPhot.yaml"):
        """Used to load the model from a saved state.

        Sets the model window to the saved value and updates all
        parameters with the saved information. This overrides the
        current parameter settings.

        Args:
          filename: The source from which to load the model parameters. Can be a string (the name of the file on disc), a dictionary (formatted as if from self.get_state), or an io.TextIOBase (a file stream to load the file from).

        """
        state = AutoPhot_Model.load(filename)
        self.name = state["name"]
        self.window = Window(**state["window"])
        for key in self.track_attrs:
            if key in state:
                setattr(self, key, state[key])
        self.parameters = Parameter_Group(self.name)
        for P in state["parameter_order"]:
            self.parameters.add_parameter(Parameter(**state["parameters"][P]))
        self.parameters.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        return state

    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters