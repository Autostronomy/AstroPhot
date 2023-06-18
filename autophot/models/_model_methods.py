from typing import Optional, Union, Dict, Tuple, Any
from copy import deepcopy

import numpy as np
import torch

from .parameter_object import Parameter
from ..utils.interpolate import _shift_Lanczos_kernel_torch, simpsons_kernel, curvature_kernel
from ..image import Model_Image, Target_Image, Window
from ..utils.operations import (
    fft_convolve_torch,
    fft_convolve_multi_torch,
    grid_integrate,
)
from .. import AP_config


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
        X, Y = Coords - center[...,None,None]
        mid = self.evaluate_model(
            X = X, Y = Y,
            image=image, parameters=parameters
        )
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        curvature = torch.nn.functional.pad(torch.nn.functional.conv2d(
            mid.view(1, 1, *mid.shape),
            kernel.view(1, 1, *kernel.shape),
            padding="valid",
        ), (1,1,1,1), mode = "replicate").squeeze()
        return mid + curvature, mid            
    elif self.sampling_mode == "trapezoid" and max(image.data.shape) >= 100:
        Coords = image.get_coordinate_corner_meshgrid()
        X, Y = Coords - center[...,None,None]
        dens = self.evaluate_model(
            X = X, Y = Y,
            image=image, parameters=parameters
        )
        kernel = torch.ones((1,1,2,2), dtype = AP_config.ap_dtype, device = AP_config.ap_device) / 4.
        trapz = torch.nn.functional.conv2d(dens.view(1,1,*dens.shape), kernel, padding="valid")
        trapz = trapz.squeeze()
        kernel = curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        curvature = torch.nn.functional.pad(torch.nn.functional.conv2d(
            trapz.view(1, 1, *trapz.shape),
            kernel.view(1, 1, *kernel.shape),
            padding="valid",
        ), (1,1,1,1), mode = "replicate").squeeze()
        return trapz + curvature, trapz
            
    Coords = image.get_coordinate_simps_meshgrid()
    X, Y = Coords - center[...,None,None]
    dens = self.evaluate_model(
        X = X, Y = Y,
        image=image, parameters=parameters
    )
    kernel = simpsons_kernel(dtype = AP_config.ap_dtype, device = AP_config.ap_device)
    mid = torch.nn.functional.conv2d(dens.view(1,1,*dens.shape), torch.ones_like(kernel) / 9, stride = 2, padding="valid") #dens[1::2,1::2]
    simps = torch.nn.functional.conv2d(dens.view(1,1,*dens.shape), kernel, stride = 2, padding="valid")
    return mid.squeeze(), simps.squeeze()

def _sample_integrate(self, deep, reference, image, parameters, center):
    if self.integrate_mode == "none":
        pass
    elif self.integrate_mode == "threshold":
        Coords = image.get_coordinate_meshgrid()
        X, Y = Coords - center[...,None, None]
        ref = torch.sum(deep) / deep.numel()
        error = torch.abs((deep - reference))
        select = error > (self.sampling_tolerance*ref)
        intdeep = grid_integrate(
            X=X[select],
            Y=Y[select],
            value = deep[select],
            compare = reference[select],
            image_header=image.header,
            eval_brightness=self.evaluate_model,
            eval_parameters=parameters,
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
            tolerance=self.sampling_tolerance,
            reference=ref,
        )
        deep[select] = intdeep
    else:
        raise ValueError(
            f"{self.name} has unknown integration mode: {self.integrate_mode}"
        )
    return deep

def _sample_convolve(self, image, shift, psf):
    if shift is not None:
        if any(np.array(psf.data.shape) < 10):
            psf_data = torch.nn.functional.pad(psf.data, (2,2,2,2))
        else:
            psf_data = psf.data
        pix_center_shift = image.world_to_pixel_delta(shift)
        LL = _shift_Lanczos_kernel_torch(
            -pix_center_shift[0],
            -pix_center_shift[1],
            2,
            AP_config.ap_dtype,
            AP_config.ap_device,
        )
        shift_psf = torch.nn.functional.conv2d(
            psf_data.view(1, 1, *psf_data.shape),
            LL.view(1, 1, *LL.shape),
            padding="valid",
        ).squeeze()
    else:
        shift_psf = psf.data
        
    if self.psf_convolve_mode == "fft":
        image.data = fft_convolve_torch(
            image.data, shift_psf / torch.sum(shift_psf), img_prepadded=True
        )
    elif self.psf_convolve_mode == "direct":
        image.data = torch.nn.functional.conv2d(
            image.data.view(1, 1, *image.data.shape),
            torch.flip(shift_psf.view(1, 1, *shift_psf.shape) / torch.sum(shift_psf), dims = (2,3)),
            padding="same",
        ).squeeze()
    else:
        raise ValueError(f"unrecognized psf_convolve_mode: {self.psf_convolve_mode}")
