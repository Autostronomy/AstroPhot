from typing import Optional, Union, Dict, Tuple, Any
from copy import deepcopy

import numpy as np
import torch

from .parameter_object import Parameter
from ..utils.decorators import ignore_numpy_warnings, default_internal
from ..utils.interpolate import (
    _shift_Lanczos_kernel_torch,
    simpsons_kernel,
    curvature_kernel,
    interp2d,
)
from ..image import Model_Image, Target_Image, Window
from ..utils.operations import (
    fft_convolve_torch,
    fft_convolve_multi_torch,
    grid_integrate,
)
from .. import AP_config


@default_internal
def angular_metric(self, X, Y, image=None, parameters=None):
    return torch.atan2(Y, X)


@default_internal
def radius_metric(self, X, Y, image=None, parameters=None):
    return torch.sqrt(
        (X) ** 2 + (Y) ** 2 + self.softening**2
    )


@classmethod
def build_parameter_specs(cls, user_specs=None):
    parameter_specs = {}
    for base in cls.__bases__:
        try:
            parameter_specs.update(base.build_parameter_specs())
        except AttributeError:
            pass
    parameter_specs.update(cls.parameter_specs)
    parameter_specs = deepcopy(parameter_specs)
    if isinstance(user_specs, dict):
        for p in user_specs:
            # If the user supplied a parameter object subclass, simply use that as is
            if isinstance(user_specs[p], Parameter):
                parameter_specs[p] = user_specs[p]
            elif isinstance(
                user_specs[p], dict
            ):  # if the user supplied parameter specifications, update the defaults
                parameter_specs[p].update(user_specs[p])
            else:
                parameter_specs[p]["value"] = user_specs[p]

    return parameter_specs


def build_parameters(self):
    for p in self.__class__._parameter_order:
        # skip if the parameter already exists
        if p in self.parameters:
            continue
        # If a parameter object is provided, simply use as-is
        if isinstance(self.parameter_specs[p], Parameter):
            self.parameters.add_parameter(self.parameter_specs[p].to())
        elif isinstance(self.parameter_specs[p], dict):
            self.parameters.add_parameter(Parameter(p, **self.parameter_specs[p]))
        else:
            raise ValueError(f"unrecognized parameter specification for {p}")


def _sample_init(self, image, parameters, center):
    if self.sampling_mode == "midpoint" and max(image.data.shape) >= 100:
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[..., None, None]
        mid = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        # convolve curvature kernel to numericall compute second derivative
        curvature = torch.nn.functional.pad(
            torch.nn.functional.conv2d(
                mid.view(1, 1, *mid.shape),
                kernel.view(1, 1, *kernel.shape),
                padding="valid",
            ),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()
        return mid + curvature, mid
    elif self.sampling_mode == "trapezoid" and max(image.data.shape) >= 100:
        Coords = image.get_coordinate_corner_meshgrid()
        X, Y = Coords - center[..., None, None]
        dens = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
        kernel = (
            torch.ones(
                (1, 1, 2, 2), dtype=AP_config.ap_dtype, device=AP_config.ap_device
            )
            / 4.0
        )
        trapz = torch.nn.functional.conv2d(
            dens.view(1, 1, *dens.shape), kernel, padding="valid"
        )
        trapz = trapz.squeeze()
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        curvature = torch.nn.functional.pad(
            torch.nn.functional.conv2d(
                trapz.view(1, 1, *trapz.shape),
                kernel.view(1, 1, *kernel.shape),
                padding="valid",
            ),
            (1, 1, 1, 1),
            mode="replicate",
        ).squeeze()
        return trapz + curvature, trapz

    Coords = image.get_coordinate_simps_meshgrid()
    X, Y = Coords - center[..., None, None]
    dens = self.evaluate_model(X=X, Y=Y, image=image, parameters=parameters)
    kernel = simpsons_kernel(dtype=AP_config.ap_dtype, device=AP_config.ap_device)
    # midpoint is just every other sample in the simpsons grid
    mid = dens[1::2, 1::2]
    simps = torch.nn.functional.conv2d(
        dens.view(1, 1, *dens.shape), kernel, stride=2, padding="valid"
    )
    return mid.squeeze(), simps.squeeze()


def _integrate_reference(self, image_data, image_header, parameters):
    return torch.sum(image_data) / image_data.numel()


def _sample_integrate(self, deep, reference, image, parameters, center):
    if self.integrate_mode == "none":
        pass
    elif self.integrate_mode == "threshold":
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[..., None, None]
        ref = self._integrate_reference(
            deep, image.header, parameters
        )  # fixme, error can be over 100% on initial sampling reference is invalid
        error = torch.abs((deep - reference))
        select = error > (self.sampling_tolerance * ref)
        intdeep = grid_integrate(
            X=X[select],
            Y=Y[select],
            image_header=image.header,
            eval_brightness=self.evaluate_model,
            eval_parameters=parameters,
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            quad_level=self.integrate_quad_level,
            gridding=self.integrate_gridding,
            max_depth=self.integrate_max_depth,
            reference=self.sampling_tolerance * ref,
        )
        deep[select] = intdeep
    else:
        raise ValueError(
            f"{self.name} has unknown integration mode: {self.integrate_mode}"
        )
    return deep


def _sample_convolve(self, image, shift, psf, shift_method="bilinear"):
    """
    image: Image object with image.data pixel matrix
    shift: the amount of shifting to do in pixel units
    psf: a PSF_Image object
    """
    if shift is not None:
        if shift_method == "bilinear":
            psf_data = torch.nn.functional.pad(psf.data, (1, 1, 1, 1))
            X, Y = torch.meshgrid(
                torch.arange(
                    psf_data.shape[1],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - shift[0],
                torch.arange(
                    psf_data.shape[0],
                    dtype=AP_config.ap_dtype,
                    device=AP_config.ap_device,
                )
                - shift[1],
                indexing="xy",
            )
            shift_psf = interp2d(psf_data, X.clone(), Y.clone())
        elif "lanczos" in shift_method:
            lanczos_order = int(shift_method[shift_method.find(":") + 1 :])
            psf_data = torch.nn.functional.pad(
                psf.data, (lanczos_order, lanczos_order, lanczos_order, lanczos_order)
            )
            LL = _shift_Lanczos_kernel_torch(
                -shift[0],
                -shift[1],
                lanczos_order,
                AP_config.ap_dtype,
                AP_config.ap_device,
            )
            shift_psf = torch.nn.functional.conv2d(
                psf_data.view(1, 1, *psf_data.shape),
                LL.view(1, 1, *LL.shape),
                padding="same",
            ).squeeze()
        else:
            raise ValueError(f"unrecognized subpixel shift method: {shift_method}")
    else:
        shift_psf = psf.data
    shift_psf = shift_psf / torch.sum(shift_psf)
    if self.psf_convolve_mode == "fft":
        image.data = fft_convolve_torch(image.data, shift_psf, img_prepadded=True)
    elif self.psf_convolve_mode == "direct":
        image.data = torch.nn.functional.conv2d(
            image.data.view(1, 1, *image.data.shape),
            torch.flip(
                shift_psf.view(1, 1, *shift_psf.shape),
                dims=(2, 3),
            ),
            padding="same",
        ).squeeze()
    else:
        raise ValueError(f"unrecognized psf_convolve_mode: {self.psf_convolve_mode}")
