from typing import Optional, Union
from copy import deepcopy

import torch

from ..param import Module, forward, Param
from ..utils.decorators import classproperty
from ..image import Window, ImageList, ModelImage, ModelImageList
from ..errors import UnrecognizedModel, InvalidWindow
from .. import AP_config
from . import func

__all__ = ("Model",)


######################################################################
class Model(Module):
    """Core class for all AstroPhot models and model like objects. This
    class defines the signatures to interact with AstroPhot models
    both for users and internal functions.

    Basic usage:

    .. code-block:: python

      import astrophot as ap

      # Create a model object
      model = ap.models.AstroPhot_Model(
          name="unique name",
          model_type="choose a model type",
          target="Target_Image object",
          window="[[xmin, xmax],[ymin,ymax]]",  # <window pixel coordinates>,
          parameters="dict of parameter specifications if desired",
      )

      # Initialize parameters that weren't set on creation
      model.initialize()

      # Fit model to target
      result = ap.fit.lm(model, verbose=1).fit()

      # Plot the model
      fig, ax = plt.subplots()
      ap.plots.model_image(fig, ax, model)
      plt.show()

      # Sample the model
      img = model()
      pixels = img.data

    AstroPhot models are one of the main ways that one interacts with
    the code, either by setting model parameters or passing models to
    other objects, one can perform a huge variety of fitting
    tasks. The subclass `Component_Model` should be thought of as the
    basic unit when constructing a model of an image while a
    `Group_Model` is a composite structure that may represent a
    complex object, a region of an image, or even a model spanning
    many images. Constructing the `Component_Model`s is where most
    work goes, these store the actual parameters that will be
    optimized. It is important to remember that a `Component_Model`
    only ever applies to a single image and a single component (star,
    galaxy, or even sub-component of one of those) in that image.

    A complex representation is made by stacking many
    `Component_Model`s together, in total this may result in a very
    large number of parameters. Trying to find starting values for all
    of these parameters can be tedious and error prone, so instead all
    built-in AstroPhot models can self initialize and find reasonable
    starting parameters for most situations. Even still one may find
    that for extremely complex fits, it is more stable to first run an
    iterative fitter before global optimization to start the models in
    better initial positions.

    Args:
        name (Optional[str]): every AstroPhot model should have a unique name
        model_type (str): a model type string can determine which kind of AstroPhot model is instantiated.
        target (Optional[Target_Image]): A Target_Image object which stores information about the image which the model is trying to fit.
        filename (Optional[str]): name of a file to load AstroPhot parameters, window, and name. The model will still need to be told its target, device, and other information
        window (Optional[Union[Window, tuple]]): A window on the target image in which the model will be optimized and evaluated. If not provided, the model will assume a window equal to the target it is fitting. The window may be formatted as (i_low, i_high, j_low, j_high) or as ((i_low, j_low), (i_high, j_high)).

    """

    _model_type = "model"
    _parameter_specs = {}
    default_uncertainty = 1e-2  # During initialization, uncertainty will be assumed 1% of initial value if no uncertainty is given
    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3  # arcsec
    _options = ("default_uncertainty", "softening")
    usable = False

    def __new__(cls, *, filename=None, model_type=None, **kwargs):
        if filename is not None:
            state = Model.load(filename)
            MODELS = Model.List_Models()
            for M in MODELS:
                if M.model_type == state["model_type"]:
                    return super(Model, cls).__new__(M)
            else:
                raise UnrecognizedModel(f"Unknown AstroPhot model type: {state['model_type']}")
        elif model_type is not None:
            MODELS = Model.List_Models()  # all_subclasses(Model)
            for M in MODELS:
                if M.model_type == model_type:
                    return super(Model, cls).__new__(M)
            else:
                raise UnrecognizedModel(f"Unknown AstroPhot model type: {model_type}")

        return super().__new__(cls)

    def __init__(self, *, name=None, target=None, window=None, mask=None, filename=None, **kwargs):
        super().__init__(name=name)
        self.target = target
        self.window = window
        self.mask = mask

        # Set any user defined options for the model
        for kwarg in list(kwargs.keys()):
            if kwarg in self.options:
                setattr(self, kwarg, kwargs.pop(kwarg))

        # Create Param objects for this Module
        parameter_specs = self.build_parameter_specs(kwargs, self.parameter_specs)
        for key in parameter_specs:
            param = Param(
                key, **parameter_specs[key], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            setattr(self, key, param)
        overload_specs = self.build_parameter_specs(kwargs, self.overload_parameter_specs)
        for key in overload_specs:
            overload = overload_specs[key].pop("overloads")
            if self[overload].value is not None:
                continue
            self[overload].value = overload_specs[key].pop("overload_function")
            param = Param(
                key, **overload_specs[key], dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            setattr(self, key, param)
            self[overload].link(key, self[key])

        self.saveattrs.update(self.options)
        self.saveattrs.add("window.extent")

        kwargs.pop("model_type", None)  # model_type is set by __new__
        if len(kwargs) > 0:
            raise TypeError(
                f"Unrecognized keyword arguments for {self.__class__.__name__}: {', '.join(kwargs.keys())}"
            )

    @classproperty
    def model_type(cls) -> str:
        collected = []
        for subcls in cls.mro():
            if subcls is object:
                continue
            mt = subcls.__dict__.get("_model_type", None)
            if mt:
                collected.append(mt)
        return " ".join(collected)

    @classproperty
    def options(cls) -> set:
        options = set()
        for subcls in cls.mro():
            if subcls is object:
                continue
            options.update(subcls.__dict__.get("_options", []))
        return options

    @classproperty
    def parameter_specs(cls) -> dict:
        """Collects all parameter specifications from the class hierarchy."""
        specs = {}
        for subcls in reversed(cls.mro()):
            if subcls is object:
                continue
            specs.update(getattr(subcls, "_parameter_specs", {}))
        return specs

    @classproperty
    def overload_parameter_specs(cls) -> dict:
        """Collects all parameter specifications from the class hierarchy."""
        specs = {}
        for subcls in reversed(cls.mro()):
            if subcls is object:
                continue
            specs.update(getattr(subcls, "_overload_parameter_specs", {}))
        return specs

    def build_parameter_specs(self, kwargs, parameter_specs) -> dict:
        parameter_specs = deepcopy(parameter_specs)

        for p in list(kwargs.keys()):
            if p not in parameter_specs:
                continue
            if isinstance(kwargs[p], dict):
                parameter_specs[p].update(kwargs.pop(p))
            else:
                parameter_specs[p]["dynamic_value"] = kwargs.pop(p)
                parameter_specs[p].pop("value", None)

        return parameter_specs

    @forward
    def gaussian_negative_log_likelihood(
        self,
        window: Optional[Window] = None,
    ) -> torch.Tensor:
        """
        Compute the negative log likelihood of the model wrt the target image in the appropriate window.
        """

        if window is None:
            window = self.window
        model = self(window=window).data
        data = self.target[window]
        weight = data.weight
        mask = data.mask
        data = data.data
        if isinstance(data, ImageList):
            nll = sum(
                torch.sum(((mo - da) ** 2 * wgt)[~ma]) / 2.0
                for mo, da, wgt, ma in zip(model, data, weight, mask)
            )
        else:
            nll = torch.sum(((model - data) ** 2 * weight)[~mask]) / 2.0

        return nll

    @forward
    def poisson_negative_log_likelihood(
        self,
        window: Optional[Window] = None,
    ) -> torch.Tensor:
        """
        Compute the negative log likelihood of the model wrt the target image in the appropriate window.
        """
        if window is None:
            window = self.window
        model = self(window=window).data
        data = self.target[window]
        mask = data.mask
        data = data.data

        if isinstance(data, ImageList):
            nll = sum(
                torch.sum((mo - da * (mo + 1e-10).log() + torch.lgamma(da + 1))[~ma])
                for mo, da, ma in zip(model, data, mask)
            )
        else:
            nll = torch.sum((model - data * (model + 1e-10).log() + torch.lgamma(data + 1))[~mask])

        return nll

    @forward
    def total_flux(self, window=None) -> torch.Tensor:
        F = self(window=window)
        return torch.sum(F.data)

    @property
    def window(self) -> Optional[Window]:
        """The window defines a region on the sky in which this model will be
        optimized and typically evaluated. Two models with
        non-overlapping windows are in effect independent of each
        other. If there is another model with a window that spans both
        of them, then they are tenuously connected.

        If not provided, the model will assume a window equal to the
        target it is fitting. Note that in this case the window is not
        explicitly set to the target window, so if the model is moved
        to another target then the fitting window will also change.

        """
        if self._window is None:
            if self.target is None:
                raise ValueError(
                    "This model has no target or window, these must be provided by the user"
                )
            return self.target.window
        return self._window

    @window.setter
    def window(self, window):
        if window is None:
            self._window = None
        elif isinstance(window, Window):
            self._window = window
        elif len(window) in [2, 4]:
            self._window = Window(window, image=self.target)
        else:
            raise InvalidWindow(f"Unrecognized window format: {str(window)}")

    @classmethod
    def List_Models(cls, usable: Optional[bool] = None, types: bool = False) -> set:
        MODELS = func.all_subclasses(cls)
        result = set()
        for model in MODELS:
            if not (model.__dict__.get("usable", False) is usable or usable is None):
                continue
            if types:
                result.add(model.model_type)
            else:
                result.add(model)
        return result

    def radius_metric(self, x, y):
        return (x**2 + y**2 + self.softening**2).sqrt()

    def angular_metric(self, x, y):
        return torch.atan2(y, x)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        super().to(dtype=dtype, device=device)

    @forward
    def __call__(
        self,
        window: Optional[Window] = None,
        **kwargs,
    ) -> Union[ModelImage, ModelImageList]:

        return self.sample(window=window, **kwargs)
