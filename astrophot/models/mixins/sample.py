from typing import Optional, Literal

import numpy as np
from torch.autograd.functional import jacobian

from ...param import forward
from ...backend_obj import backend, ArrayLike
from ... import config
from ...image import Image, Window, JacobianImage
from .. import func
from ...errors import SpecificationConflict


class SampleMixin:
    """
    **Options:**
    -    `sampling_mode`: The method used to sample the model in image pixels. Options are:
            - `auto`: Automatically choose the sampling method based on the image size.
            - `midpoint`: Use midpoint sampling, evaluate the brightness at the center of each pixel.
            - `simpsons`: Use Simpson's rule for sampling integrating each pixel.
            - `quad:x`: Use quadrature sampling with order x, where x is a positive integer to integrate each pixel.
    -    `jacobian_maxparams`: The maximum number of parameters before the Jacobian will be broken into smaller chunks. This is helpful for limiting the memory requirements to build a model.
    -    `jacobian_maxpixels`: The maximum number of pixels before the Jacobian will be broken into smaller chunks. This is helpful for limiting the memory requirements to build a model.
    -    `integrate_mode`: The method used to select pixels to integrate further where the model varies significantly. Options are:
            - `none`: No extra integration is performed (beyond the sampling_mode).
            - `bright`: Select the brightest pixels for further integration.
            - `threshold`: Select pixels which show signs of significant higher order derivatives.
    -    `integrate_tolerance`: The tolerance for selecting a pixel in the integration method. This is the total flux fraction that is integrated over the image.
    -    `integrate_fraction`: The fraction of the pixels to super sample during integration.
    -    `integrate_max_depth`: The maximum depth of the integration method.
    -    `integrate_gridding`: The gridding used for the integration method to super-sample a pixel at each iteration.
    -    `integrate_quad_order`: The order of the quadrature used for the integration method on the super sampled pixels.
    """

    # Method for initial sampling of model
    sampling_mode = "auto"  # auto (choose based on image size), midpoint, simpsons, quad:x (where x is a positive integer)

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_maxparams = 10
    jacobian_maxpixels = 1000**2
    integrate_mode = "bright"  # none, bright, curvature
    integrate_fraction = 0.05  # fraction of the pixels to super sample
    integrate_max_depth = 2
    integrate_gridding = 5
    integrate_quad_order = 3

    _options = (
        "sampling_mode",
        "jacobian_maxparams",
        "jacobian_maxpixels",
        "integrate_mode",
        "integrate_fraction",
        "integrate_max_depth",
        "integrate_gridding",
        "integrate_quad_order",
    )

    @forward
    def _bright_integrate(self, sample: ArrayLike, image: Image) -> ArrayLike:
        i, j = image.pixel_center_meshgrid()
        sample = func.bright_integrate(
            sample,
            i,
            j,
            lambda i, j: self.brightness(*image.pixel_to_plane(i, j)),
            scale=image.base_scale,
            bright_frac=self.integrate_fraction,
            quad_order=self.integrate_quad_order,
            gridding=self.integrate_gridding,
            max_depth=self.integrate_max_depth,
        )
        return sample

    @forward
    def _curvature_integrate(self, sample: ArrayLike, image: Image) -> ArrayLike:
        i, j = image.pixel_center_meshgrid()
        kernel = func.curvature_kernel(config.DTYPE, config.DEVICE)
        curvature = (
            backend.abs(
                backend.pad(
                    backend.conv2d(
                        sample.reshape(1, 1, *sample.shape),
                        kernel.reshape(1, 1, *kernel.shape),
                        padding="valid",
                    ),
                    (0, 0, 0, 0, 1, 1, 1, 1),
                    mode="replicate",
                )
            )
            .squeeze(0)
            .squeeze(0)
        )
        N = max(1, int(np.prod(image.data.shape) * self.integrate_fraction))
        select = backend.topk(curvature.flatten(), N)[1]

        sample_flat = sample.flatten()
        sample_flat = backend.fill_at_indices(
            sample_flat,
            select,
            func.recursive_quad_integrate(
                i.flatten()[select],
                j.flatten()[select],
                lambda i, j: self.brightness(*image.pixel_to_plane(i, j)),
                scale=image.base_scale,
                curve_frac=self.integrate_fraction,
                quad_order=self.integrate_quad_order,
                gridding=self.integrate_gridding,
                max_depth=self.integrate_max_depth,
            ),
        )
        return sample_flat.reshape(sample.shape)

    @forward
    def sample_image(self, image: Image) -> ArrayLike:
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
        if self.integrate_mode == "curvature":
            sample = self._curvature_integrate(sample, image)
        elif self.integrate_mode == "bright":
            sample = self._bright_integrate(sample, image)
        elif self.integrate_mode != "none":
            raise SpecificationConflict(
                f"Unknown integrate mode {self.integrate_mode} for model {self.name}"
            )
        return sample

    def _jacobian(
        self, window: Window, params_pre: ArrayLike, params: ArrayLike, params_post: ArrayLike
    ) -> ArrayLike:
        # return jacfwd( # this should be more efficient, but the trace overhead is too high
        #     lambda x: self.sample(
        #         window=window, params=torch.cat((params_pre, x, params_post), dim=-1)
        #     ).data
        # )(params)
        return backend.jacobian(
            lambda x: self.sample(
                window=window, params=backend.concatenate((params_pre, x, params_post), dim=-1)
            ).data,
            params,
        )

    def jacobian(
        self,
        window: Optional[Window] = None,
        pass_jacobian: Optional[JacobianImage] = None,
        params: Optional[ArrayLike] = None,
    ) -> JacobianImage:
        if window is None:
            window = self.window

        if pass_jacobian is None:
            jac_img = self.target[window].jacobian_image(
                parameters=self.build_params_array_identities()
            )
        else:
            jac_img = pass_jacobian

        # No dynamic params
        if params is None:
            params = self.build_params_array()
        if params.shape[-1] == 0:
            return jac_img

        # handle large images
        n_pixels = np.prod(window.shape)
        if n_pixels > self.jacobian_maxpixels:
            for chunk in window.chunk(self.jacobian_maxpixels):
                jac_img = self.jacobian(window=chunk, pass_jacobian=jac_img, params=params)
            return jac_img

        identities = self.build_params_array_identities()
        if len(jac_img.match_parameters(identities)[0]) == 0:
            return jac_img

        target = self.target[window]
        if len(params) > self.jacobian_maxparams:  # handle large number of parameters
            chunksize = len(params) // self.jacobian_maxparams + 1
            for i in range(0, len(params), chunksize):
                params_pre = params[:i]
                params_chunk = params[i : i + chunksize]
                params_post = params[i + chunksize :]
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
        params: Optional[ArrayLike] = None,
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
    ) -> ArrayLike:
        """Compute the gradient of the model with respect to its parameters."""
        if window is None:
            window = self.window

        jacobian_image = self.jacobian(window=window, params=params)

        data = self.target[window].data
        model = self.sample(window=window).data
        if likelihood == "gaussian":
            weight = self.target[window].weight
            gradient = backend.sum(
                jacobian_image.data * ((data - model) * weight)[..., None], dim=(0, 1)
            )
        elif likelihood == "poisson":
            gradient = backend.sum(
                jacobian_image.data * (1 - data / model)[..., None],
                dim=(0, 1),
            )

        return gradient
