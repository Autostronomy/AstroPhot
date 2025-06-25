from typing import Optional, Literal

import numpy as np
from torch.autograd.functional import jacobian
import torch
from torch import Tensor

from ...param import forward
from ... import AP_config
from ...image import Image, Window, JacobianImage
from .. import func
from ...errors import SpecificationConflict


class SampleMixin:
    # Method for initial sampling of model
    sampling_mode = "auto"  # auto (choose based on image size), midpoint, simpsons, quad:x (where x is a positive integer)

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_maxparams = 10
    jacobian_maxpixels = 1000**2
    integrate_mode = "threshold"  # none, threshold
    integrate_tolerance = 1e-3  # total flux fraction
    integrate_max_depth = 3
    integrate_gridding = 5
    integrate_quad_order = 3

    _options = (
        "sampling_mode",
        "jacobian_maxparams",
        "jacobian_maxpixels",
        "psf_subpixel_shift",
        "integrate_mode",
        "integrate_tolerance",
        "integrate_max_depth",
        "integrate_gridding",
        "integrate_quad_order",
    )

    def shift_kernel(self, shift):
        if self.psf_subpixel_shift == "bilinear":
            return func.bilinear_kernel(shift[0], shift[1])
        elif self.psf_subpixel_shift.startswith("lanczos:"):
            order = int(self.psf_subpixel_shift.split(":")[1])
            return func.lanczos_kernel(shift[0], shift[1], order)
        elif self.psf_subpixel_shift == "none":
            return torch.tensor(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                dtype=AP_config.ap_dtype,
                device=AP_config.ap_device,
            )
        else:
            raise SpecificationConflict(
                f"Unknown PSF subpixel shift mode {self.psf_subpixel_shift} for model {self.name}"
            )

    @forward
    def _sample_integrate(self, sample, image: Image):
        i, j = image.pixel_center_meshgrid()
        kernel = func.curvature_kernel(AP_config.ap_dtype, AP_config.ap_device)
        curvature = (
            torch.nn.functional.pad(
                torch.nn.functional.conv2d(
                    sample.view(1, 1, *sample.shape),
                    kernel.view(1, 1, *kernel.shape),
                    padding="valid",
                ),
                (1, 1, 1, 1),
                mode="replicate",
            )
            .squeeze(0)
            .squeeze(0)
            .abs()
        )
        total_est = torch.sum(sample)
        threshold = total_est * self.integrate_tolerance
        select = curvature > threshold

        sample[select] = func.recursive_quad_integrate(
            i[select],
            j[select],
            lambda i, j: self.brightness(*image.pixel_to_plane(i, j)),
            threshold=threshold,
            quad_order=self.integrate_quad_order,
            gridding=self.integrate_gridding,
            max_depth=self.integrate_max_depth,
        )
        return sample

    @forward
    def sample_image(self, image: Image):
        if self.sampling_mode == "auto":
            N = np.prod(image.data.shape)
            if N <= 100:
                sampling_mode = "quad:5"
            elif N <= 10000:
                sampling_mode = "simpsons"
            else:
                sampling_mode = "midpoint"
        else:
            sampling_mode = self.sampling_mode
        if sampling_mode == "midpoint":
            x, y = image.coordinate_center_meshgrid()
            res = self.brightness(x, y)
            sample = func.pixel_center_integrator(res)
        elif sampling_mode == "simpsons":
            x, y = image.coordinate_simpsons_meshgrid()
            res = self.brightness(x, y)
            sample = func.pixel_simpsons_integrator(res)
        elif sampling_mode.startswith("quad:"):
            order = int(self.sampling_mode.split(":")[1])
            i, j, w = image.pixel_quad_meshgrid(order=order)
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            sample = func.pixel_quad_integrator(res, w)
        else:
            raise SpecificationConflict(
                f"Unknown sampling mode {self.sampling_mode} for model {self.name}"
            )
        if self.integrate_mode == "threshold":
            sample = self._sample_integrate(sample, image)
        return sample

    def _jacobian(self, window: Window, params_pre: Tensor, params: Tensor, params_post: Tensor):
        return jacobian(
            lambda x: self.sample(
                window=window, params=torch.cat((params_pre, x, params_post), dim=-1)
            ).data.value,
            params,
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

    def jacobian(
        self,
        window: Optional[Window] = None,
        pass_jacobian: Optional[JacobianImage] = None,
        params: Optional[Tensor] = None,
    ):
        if window is None:
            window = self.window

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.build_params_array_identities()
            )
        else:
            jac_img = pass_jacobian

        # No dynamic params
        if len(self.build_params_list()) == 0:
            return jac_img

        # handle large images
        n_pixels = np.prod(window.shape)
        if n_pixels > self.jacobian_maxpixels:
            for chunk in window.chunk(self.jacobian_maxpixels):
                self.jacobian(window=chunk, pass_jacobian=jac_img, params=params)
            return jac_img

        if params is None:
            params = self.build_params_array()
        identities = self.build_params_array_identities()
        target = self.target[window]
        if len(params) > self.jacobian_maxparams:  # handle large number of parameters
            chunksize = len(params) // self.jacobian_maxparams + 1
            for i in range(chunksize, len(params), chunksize):
                params_pre = params[:i]
                params_post = params[i + chunksize :]
                params_chunk = params[i : i + chunksize]
                jac_chunk = self._jacobian(window, params_pre, params_chunk, params_post)
                jac_img += target.jacobian_image(
                    parameters=identities[i : i + chunksize],
                    data=jac_chunk,
                )
        else:
            jac = self._jacobian(window, params[:0], params, params[0:0])
            jac_img += target.jacobian_image(parameters=identities, data=jac)

        return jac_img

    def gradient(
        self,
        window: Optional[Window] = None,
        params: Optional[Tensor] = None,
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
    ):
        """Compute the gradient of the model with respect to its parameters."""
        if window is None:
            window = self.window

        jacobian_image = self.jacobian(window=window, params=params)

        data = self.target[window].data.value
        model = self.sample(window=window).data.value
        if likelihood == "gaussian":
            weight = self.target[window].weight
            gradient = torch.sum(
                jacobian_image.data.value * ((data - model) * weight).unsqueeze(-1), dim=(0, 1)
            )
        elif likelihood == "poisson":
            gradient = torch.sum(
                jacobian_image.data.value * (1 - data / model).unsqueeze(-1),
                dim=(0, 1),
            )

        return gradient
