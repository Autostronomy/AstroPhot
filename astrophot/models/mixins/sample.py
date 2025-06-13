from typing import Optional, Literal

import numpy as np
from caskade import forward
from torch.autograd.functional import jacobian
import torch
from torch import Tensor

from ... import AP_config
from ...image import Image, Window, Jacobian_Image
from .. import func
from ...errors import SpecificationConflict


class SampleMixin:
    # Method for initial sampling of model
    sampling_mode = "auto"  # auto (choose based on image size), midpoint, simpsons, quad:x (where x is a positive integer)

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_maxparams = 10
    jacobian_maxpixels = 1000**2

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
            i, j = func.pixel_center_meshgrid(image.shape, AP_config.ap_dtype, AP_config.ap_device)
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_center_integrator(res)
        elif sampling_mode == "simpsons":
            i, j = func.pixel_simpsons_meshgrid(
                image.shape, AP_config.ap_dtype, AP_config.ap_device
            )
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_simpsons_integrator(res)
        elif sampling_mode.startswith("quad:"):
            order = int(self.sampling_mode.split(":")[1])
            i, j, w = func.pixel_quad_meshgrid(
                image.shape, AP_config.ap_dtype, AP_config.ap_device, order=order
            )
            x, y = image.pixel_to_plane(i, j)
            res = self.brightness(x, y)
            return func.pixel_quad_integrator(res, w)
        raise SpecificationConflict(
            f"Unknown sampling mode {self.sampling_mode} for model {self.name}"
        )

    def build_params_array_identities(self):
        identities = []
        for param in self.dynamic_params:
            numel = max(1, np.prod(param.shape))
            for i in range(numel):
                identities.append(f"{id(param)}_{i}")
        return identities

    def _jacobian(self, window: Window, params_pre: Tensor, params: Tensor, params_post: Tensor):
        return jacobian(
            lambda x: self.sample(
                window=window, params=torch.cat((params_pre, x, params_post), dim=-1)
            ).data,
            params,
            strategy="forward-mode",
            vectorize=True,
            create_graph=False,
        )

    def jacobian(
        self,
        window: Optional[Window] = None,
        pass_jacobian: Optional[Jacobian_Image] = None,
        params: Optional[Tensor] = None,
    ):
        if window is None:
            window = self.window

        if params is not None:
            self.fill_dynamic_params(params)

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
                self.jacobian(window=chunk, pass_jacobian=jac_img)
            return jac_img

        params = self.build_params_array()
        identities = self.build_params_array_identities()
        if len(params) > self.jacobian_maxparams:  # handle large number of parameters
            chunksize = len(params) // self.jacobian_maxparams + 1
            for i in range(chunksize, len(params), chunksize):
                params_pre = params[:i]
                params_post = params[i + chunksize :]
                params_chunk = params[i : i + chunksize]
                jac_chunk = self._jacobian(window, params_pre, params_chunk, params_post)
                jac_img += self.target[window].jacobian_image(
                    parameters=identities[i : i + chunksize],
                    data=jac_chunk,
                )
        else:
            jac = self._jacobian(window, params[:0], params, params[0:0])
            jac_img += self.target[window].jacobian_image(parameters=identities, data=jac)

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
