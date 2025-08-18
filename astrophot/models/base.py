from typing import Optional, Union
from copy import deepcopy

import numpy as np

from caskade import Param as CParam
from ..param import Module, forward, Param
from ..utils.decorators import classproperty
from ..image import Window, ImageList, ModelImage, ModelImageList
from ..errors import UnrecognizedModel, InvalidWindow
from .. import config
from ..backend_obj import backend, ArrayLike
from . import func

__all__ = ("Model",)


######################################################################
class Model(Module):
    """Base class for all AstroPhot models."""

    _model_type = "model"
    _parameter_specs = {}
    # Softening length used for numerical stability and/or integration stability to avoid discontinuities (near R=0)
    softening = 1e-3  # arcsec
    _options = ("softening",)
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
            param = Param(key, **parameter_specs[key], dtype=config.DTYPE, device=config.DEVICE)
            setattr(self, key, param)

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
            if isinstance(parameter_specs[p].get("dynamic_value", None), CParam) or callable(
                parameter_specs[p].get("dynamic_value", None)
            ):
                parameter_specs[p]["value"] = parameter_specs[p]["dynamic_value"]
                parameter_specs[p].pop("dynamic_value", None)

        return parameter_specs

    @forward
    def gaussian_log_likelihood(
        self,
        window: Optional[Window] = None,
    ) -> ArrayLike:
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
        if isinstance(data, tuple):
            nll = 0.5 * sum(
                backend.sum(((da - mo) ** 2 * wgt)[~ma])
                for mo, da, wgt, ma in zip(model, data, weight, mask)
            )
        else:
            nll = 0.5 * backend.sum(((data - model) ** 2 * weight)[~mask])

        return -nll

    @forward
    def poisson_log_likelihood(
        self,
        window: Optional[Window] = None,
    ) -> ArrayLike:
        """
        Compute the negative log likelihood of the model wrt the target image in the appropriate window.
        """
        if window is None:
            window = self.window
        model = self(window=window).data
        data = self.target[window]
        mask = data.mask
        data = data.data

        if isinstance(data, tuple):
            nll = sum(
                backend.sum((mo - da * backend.log(mo + 1e-10) + backend.lgamma(da + 1))[~ma])
                for mo, da, ma in zip(model, data, mask)
            )
        else:
            nll = backend.sum(
                (model - data * backend.log(model + 1e-10) + backend.lgamma(data + 1))[~mask]
            )

        return -nll

    def hessian(self, likelihood="gaussian"):
        if likelihood == "gaussian":
            return backend.hessian(self.gaussian_log_likelihood)(self.build_params_array())
        elif likelihood == "poisson":
            return backend.hessian(self.poisson_log_likelihood)(self.build_params_array())
        else:
            raise ValueError(f"Unknown likelihood type: {likelihood}")

    def total_flux(self, window=None) -> ArrayLike:
        F = self(window=window)
        return backend.sum(F.data)

    def total_flux_uncertainty(self, window=None) -> ArrayLike:
        jac = self.jacobian(window=window).flatten("data")
        dF = backend.sum(jac, dim=0)  # VJP for sum(total_flux)
        current_uncertainty = self.build_params_array_uncertainty()
        return backend.sqrt(backend.sum((dF * current_uncertainty) ** 2))

    def total_magnitude(self, window=None) -> ArrayLike:
        """Compute the total magnitude of the model in the given window."""
        F = self.total_flux(window=window)
        return -2.5 * backend.log10(F) + self.target.zeropoint

    def total_magnitude_uncertainty(self, window=None) -> ArrayLike:
        """Compute the uncertainty in the total magnitude of the model in the given window."""
        F = self.total_flux(window=window)
        dF = self.total_flux_uncertainty(window=window)
        return 2.5 * (dF / F) / np.log(10)

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

    @forward
    def radius_metric(self, x, y):
        return backend.sqrt(x**2 + y**2 + self.softening**2)

    @forward
    def angular_metric(self, x, y):
        return backend.arctan2(y, x)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = config.DTYPE
        if device is None:
            device = config.DEVICE
        super().to(dtype=dtype, device=device)

    @forward
    def __call__(
        self,
        window: Optional[Window] = None,
        **kwargs,
    ) -> Union[ModelImage, ModelImageList]:

        return self.sample(window=window, **kwargs)
