import torch

from .core_model import AstroPhot_Model
from ..utils.decorators import default_internal


__all__ = ["PSF_Model"]


class PSF_Model(AstroPhot_Model):
    """Prototype point source (typically a star) model, to be subclassed
    by other point source models which define specific
    behavior. 

    """

    # Specifications for the model parameters including units, value, uncertainty, limits, locked, and cyclic
    parameter_specs = {}
    # Fixed order of parameters for all methods that interact with the list of parameters
    _parameter_order = ()
    model_type = f"psf {AstroPhot_Model.model_type}"
    useable = False
    # Method for initial sampling of model
    sampling_mode = "midpoint"  # midpoint, trapezoid, simpson

    # Level to which each pixel should be evaluated
    sampling_tolerance = 1e-2

    # Integration scope for model
    integrate_mode = "threshold"  # none, threshold, full*

    # Maximum recursion depth when performing sub pixel integration
    integrate_max_depth = 3

    # Amount by which to subdivide pixels when doing recursive pixel integration
    integrate_gridding = 5

    # The initial quadrature level for sub pixel integration. Please always choose an odd number 3 or higher
    integrate_quad_level = 3

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_chunksize = 10

    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3

    # Parameters which are treated specially by the model object and should not be updated directly when initializing
    special_kwargs = ["parameters", "filename", "model_type"]
    track_attrs = [
        "psf_mode",
        "psf_convolve_mode",
        "sampling_mode",
        "sampling_tolerance",
        "integrate_mode",
        "integrate_max_depth",
        "integrate_gridding",
        "integrate_quad_level",
        "jacobian_chunksize",
        "softening",
    ]

    def __init__(self, *, name=None, **kwargs):
        self._target_identity = None
        super().__init__(name=name,**kwargs)

        # Set any user defined attributes for the model
        for kwarg in kwargs:  # fixme move to core model?
            # Skip parameters with special behaviour
            if kwarg in self.special_kwargs:
                continue
            # Set the model parameter
            setattr(self, kwarg, kwargs[kwarg])

        # If loading from a file, get model configuration then exit __init__
        if "filename" in kwargs:
            self.load(kwargs["filename"], new_name = name)
            return

        self.parameter_specs = self.build_parameter_specs(
            kwargs.get("parameters", None)
        )
        with torch.no_grad():
            self.build_parameters()
            if isinstance(kwargs.get("parameters", None), torch.Tensor):
                self.parameters.value = kwargs["parameters"]
        assert torch.all(self.window.reference_imageij == torch.zeros_like(self.window.reference_imageij)), "PSF models must always be centered at (0,0)"
        assert torch.all(self.window.reference_imagexy == torch.zeros_like(self.window.reference_imagexy)), "PSF models must always be centered at (0,0)"

    # Initialization functions
    ######################################################################
    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(
        self,
        target: Optional["Target_Image"] = None,
        parameters: Optional[Parameter_Node] = None,
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
    
    # Fit loop functions
    ######################################################################
    def evaluate_model(
        self,
        X: Optional[torch.Tensor] = None,
        Y: Optional[torch.Tensor] = None,
        image: Optional["Image"] = None,
        parameters: "Parameter_Node" = None,
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
            X, Y = Coords - parameters["center"].value[..., None, None]
        return torch.zeros_like(X)  # do nothing in base model

    def sample(
        self,
        image: Optional["Image"] = None,
        window: Optional[Window] = None,
        parameters: Optional[Parameter_Node] = None,
    ):
        """Evaluate the model on the space covered by an image object. This
        function properly calls integration methods. This should not
        be overloaded except in special cases.

        This function is designed to compute the model on a given
        image or within a specified window. It takes care of sub-pixel
        sampling, recursive integration for high curvature regions,
        and proper alignment of the computed model with the original
        pixel grid. The final model is then added to the requested
        image.

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

        if "window" in self.psf_mode:
            raise NotImplementedError("PSF convolution in sub-window not available yet")

        # Create an image to store pixel samples
        working_image = Model_Image(
            pixelscale=image.pixelscale, window=working_window
        )
        # Evaluate the model on the image
        reference, deep = self._sample_init(
            image=working_image,
            parameters=parameters,
            center=parameters["center"].value,
        )
        # Super-resolve and integrate where needed
        deep = self._sample_integrate(
            deep,
            reference,
            working_image,
            parameters,
            center=parameters["center"].value,
        )
        # Add the sampled/integrated pixels to the requested image
        working_image.data += deep

        if self.mask is not None:
            working_image.data = working_image.data * torch.logical_not(self.mask)

        image += working_image

        return image

    @torch.no_grad()
    def jacobian(
        self,
        parameters: Optional[torch.Tensor] = None,
        as_representation: bool = False,
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
        if torch.sum(self.parameters.vector_mask()) == 0 or window.overlap_frac(self.window) <= 0:
            return self.target[window].jacobian_image()

        # Set the parameters if provided and check the size of the parameter list
        if parameters is not None:
            if as_representation:
                self.parameters.vector_set_representation(parameters)
            else:
                self.parameters.vector_set_values(parameters)
        if torch.sum(self.parameters.vector_mask()) > self.jacobian_chunksize:
            return self._chunk_jacobian(
                as_representation=as_representation,
                window=window,
                **kwargs,
            )

        # Compute the jacobian
        full_jac = jacobian(
            lambda P: self(
                image=None,
                parameters=P,
                as_representation=as_representation,
                window=window,
            ).data,
            self.parameters.vector_representation().detach() if as_representation else self.parameters.vector_values().detach(),
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

        # Store the jacobian as a Jacobian_Image object
        jac_img = self.target[window].jacobian_image(
            parameters=self.parameters.vector_identities(),
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
        pids = self.parameters.vector_identities()
        jac_img = self.target[window].jacobian_image(
            parameters=pids,
        )

        for ichunk in range(0, len(pids), self.jacobian_chunksize):
            mask = torch.zeros(len(pids), dtype = torch.bool, device = AP_config.ap_device)
            mask[ichunk:ichunk+self.jacobian_chunksize] = True
            with Param_Mask(self.parameters, mask):
                jac_img += self.jacobian(
                    parameters=None,
                    as_representation=as_representation,
                    window=window,
                    **kwargs,
                )

        return jac_img

    @property
    def target(self):
        try:
            return self._target
        except AttributeError:
            return None

    @target.setter
    def target(self, tar):
        assert tar is None or isinstance(tar, Target_Image)

        # If a target image list is assigned, pick out the target appropriate for this model
        if isinstance(tar, Target_Image_List) and self._target_identity is not None:
            for subtar in tar:
                if subtar.identity == self._target_identity:
                    usetar = subtar
                    break
            else:
                raise KeyError(
                    f"Could not find target in Target_Image_List with matching identity to {self.name}: {self._target_identity}"
                )
        else:
            usetar = tar

        self._target = usetar

        # Remember the target identity to use
        try:
            self._target_identity = self._target.identity
        except AttributeError:
            pass

    def get_state(self, save_params = True):
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
        if save_params:
            state["parameters"] = self.parameters.get_state()
        state["target_identity"] = self._target_identity
        for key in self.track_attrs:
            if getattr(self, key) != getattr(self.__class__, key):
                state[key] = getattr(self, key)
        return state

    def load(self, filename: Union[str, dict, io.TextIOBase] = "AstroPhot.yaml", new_name = None):
        """Used to load the model from a saved state.

        Sets the model window to the saved value and updates all
        parameters with the saved information. This overrides the
        current parameter settings.

        Args:
          filename: The source from which to load the model parameters. Can be a string (the name of the file on disc), a dictionary (formatted as if from self.get_state), or an io.TextIOBase (a file stream to load the file from).

        """
        state = AstroPhot_Model.load(filename)
        if new_name is None:
            new_name = state["name"]
        self.name = new_name
        # Use window saved state to initialize model window
        self.window = Window(**state["window"])
        # reassign target in case a target list was given
        self._target_identity = state["target_identity"]
        self.target = self.target
        # Set any attributes which were not default
        for key in self.track_attrs:
            if key in state:
                setattr(self, key, state[key])
        # Load the parameter group, this is handled by the parameter group object
        if isinstance(state["parameters"], Parameter_Node):
            self.parameters = state["parameters"]
        else:
            self.parameters = Parameter_Node(self.name, state=state["parameters"])
        # Move parameters to the appropriate device and dtype
        self.parameters.to(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        return state

    # Extra background methods for the basemodel
    ######################################################################
    from ._model_methods import radius_metric
    from ._model_methods import angular_metric
    from ._model_methods import _sample_init
    from ._model_methods import _sample_integrate
    from ._model_methods import _integrate_reference
    from ._model_methods import build_parameter_specs
    from ._model_methods import build_parameters
