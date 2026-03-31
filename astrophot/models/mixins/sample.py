from typing import Optional, Literal

import numpy as np

from ...param import forward
from ...backend_obj import backend, ArrayLike
from ... import config
from ...image import JacobianImage
from .. import func
from ...errors import SpecificationConflict
from ...utils.integration import quad_table


class SampleMixin:
    """
    :param sampling_mode: The method used to sample the model in image pixels. Options are: `auto`: Automatically choose the sampling method based on the image size (default). `midpoint`: Use midpoint sampling, evaluate the brightness at the center of each pixel. `simpsons`: Use Simpson's rule for sampling integrating each pixel. `upsample:x` upsample the pixel in a regular grid of size x (odd positive integer), generally less accurate than quad:x. `quad:x`: Use quadrature sampling with order x, where x is an odd positive integer to integrate each pixel.
    :param integrate_mode: The method used to select pixels to integrate further where the model varies significantly. Options are: `none`: No extra integration is performed (beyond the sampling_mode). `bright`: Select the brightest pixels for further integration (default). `threshold`: Select pixels which show signs of significant higher order derivatives.
    :param integrate_fraction: The fraction of the pixels to super sample during integration (default: 0.05).
    :param integrate_max_depth: The maximum depth of the integration method (default: 2).
    :param integrate_gridding: The gridding used for the integration method to super-sample a pixel at each iteration (default: 5).
    :param integrate_quad_order: The order of the quadrature used for the integration method on the super sampled pixels (default: 3).
    """

    integrate_fraction = 0.05  # fraction of the pixels to super sample
    integrate_max_depth = 2
    integrate_gridding = 5
    integrate_quad_order = 3

    _options = (
        "sampling_mode",
        "integrate_mode",
        "integrate_fraction",
        "integrate_max_depth",
        "integrate_gridding",
        "integrate_quad_order",
    )

    def __init__(self, *args, sampling_mode="auto", integrate_mode="bright", **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_mode = sampling_mode
        self.integrate_mode = integrate_mode

    @forward
    def _bright_integrate(
        self,
        sample: ArrayLike,
        i: ArrayLike,
        j: ArrayLike,
        upsample: int,
        pixel_brightness: callable,
    ) -> ArrayLike:
        sample = func.bright_integrate(
            sample,
            i,
            j,
            pixel_brightness,
            scale=self.target.base_scale / upsample,
            bright_frac=self.integrate_fraction,
            quad_order=self.integrate_quad_order,
            gridding=self.integrate_gridding,
            max_depth=self.integrate_max_depth,
        )
        return sample

    @forward
    def _curvature_integrate(
        self,
        sample: ArrayLike,
        i: ArrayLike,
        j: ArrayLike,
        upsample: int,
        pixel_brightness: callable,
    ) -> ArrayLike:
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
        N = max(1, int(np.prod(i.shape) * self.integrate_fraction))
        select = backend.topk(curvature.flatten(), N)[1]

        sample_flat = sample.flatten()
        sample_flat = backend.fill_at_indices(
            sample_flat,
            select,
            func.recursive_quad_integrate(
                i.flatten()[select],
                j.flatten()[select],
                pixel_brightness,
                scale=self.target.base_scale / upsample,
                curve_frac=self.integrate_fraction,
                quad_order=self.integrate_quad_order,
                gridding=self.integrate_gridding,
                max_depth=self.integrate_max_depth,
            ),
        )
        return sample_flat.reshape(sample.shape)

    @property
    def sampling_mode(self):
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, sampling_mode):
        if sampling_mode == "auto":
            sampling_mode = "midpoint"
            try:
                N = np.prod(self.window.shape)
                if N <= 100:
                    sampling_mode = "quad:5"
                elif N <= 10000:
                    sampling_mode = "simpsons"
            except:
                pass
        if sampling_mode == "midpoint":
            self._pixel_meshgridder = lambda im, w, p, u: im.pixel_center_meshgrid(w, p, u)
            self._pixel_integrator = func.pixel_center_integrator
            self._pixel_center_finder = lambda i, j: (i, j)
        elif sampling_mode == "simpsons":
            self._pixel_meshgridder = lambda im, w, p, u: im.pixel_simpsons_meshgrid(w, p, u)
            self._pixel_integrator = func.pixel_simpsons_integrator
            self._pixel_center_finder = lambda i, j: (i[1:-1:2, 1:-1:2], j[1:-1:2, 1:-1:2])
        elif sampling_mode.startswith("quad:"):
            order = int(sampling_mode.split(":")[1])
            self._pixel_meshgridder = lambda im, w, p, u: im.pixel_quad_meshgrid(
                w, p, u, order=order
            )[:2]
            _, _, w = quad_table(order, config.DTYPE, config.DEVICE)
            w = w.flatten()
            self._pixel_integrator = lambda z: func.pixel_quad_integrator(z, w)
            self._pixel_center_finder = lambda i, j: (i[..., order**2 // 2], j[..., order**2 // 2])
        elif sampling_mode.startswith("upsample:"):
            upsample = int(sampling_mode.split(":")[1])
            if upsample % 2 != 1:
                raise SpecificationConflict(
                    f"Upsample factor for 'sample_mode' must be an odd integer, got {upsample} for model {self.name}"
                )
            self._pixel_meshgridder = lambda im, w, p, u: im.pixel_center_meshgrid(
                w, upsample * p, upsample * u
            )
            self._pixel_integrator = lambda z: func.downsample_mean(z, upsample)
            self._pixel_center_finder = lambda i, j: (
                i[upsample // 2 :: upsample, upsample // 2 :: upsample],
                j[upsample // 2 :: upsample, upsample // 2 :: upsample],
            )
        else:
            raise SpecificationConflict(
                f"Unknown sampling mode {sampling_mode} for model {self.name}"
            )
        self._sampling_mode = sampling_mode

    @property
    def integrate_mode(self):
        return self._integrate_mode

    @integrate_mode.setter
    def integrate_mode(self, integrate_mode):
        if integrate_mode == "bright":
            self._adaptive_integrator = self._bright_integrate
        elif integrate_mode == "curvature":
            self._adaptive_integrator = self._curvature_integrate
        elif integrate_mode == "none":
            self._adaptive_integrator = lambda z, *a, **kw: z
        else:
            raise SpecificationConflict(
                f"Unknown integrate mode {integrate_mode} for model {self.name}"
            )
        self._integrate_mode = integrate_mode


class GradMixin:
    """
    :param jacobian_maxparams: The maximum number of parameters before the Jacobian will be broken into smaller chunks to reduce memory consumption (int, default: 10).
    """

    # Maximum size of parameter list before jacobian will be broken into smaller chunks, this is helpful for limiting the memory requirements to build a model, lower jacobian_chunksize is slower but uses less memory
    jacobian_maxparams = 10

    _options = ("jacobian_maxparams",)

    def _jacobian(
        self,
        params_pre: ArrayLike,
        params: ArrayLike,
        params_post: ArrayLike,
    ) -> ArrayLike:
        # return jacfwd( # this should be more efficient, but the trace overhead is too high
        #     lambda x: self.sample(
        #         window=window, params=torch.cat((params_pre, x, params_post), dim=-1)
        #     ).data
        # )(params)
        return backend.jacobian(
            lambda x: self(params=backend.concatenate((params_pre, x, params_post), dim=-1))._data,
            params,
        )

    def jacobian(
        self,
        pass_jacobian: Optional[JacobianImage] = None,
        params: Optional[ArrayLike] = None,
    ) -> JacobianImage:

        if pass_jacobian is None:
            jac_img = self.target[self.window].jacobian_image(
                parameters=self.build_params_array_identities()
            )
        else:
            jac_img = pass_jacobian

        # No dynamic params
        if params is None:
            params = self.get_values()
        if len(params.shape) == 0 or params.shape[-1] == 0:
            return jac_img

        identities = self.build_params_array_identities()
        if len(jac_img.match_parameters(identities)[0]) == 0:
            return jac_img

        target = self.target[self.window]
        if len(params) > self.jacobian_maxparams:  # handle large number of parameters
            chunksize = len(params) // self.jacobian_maxparams + 1
            for i in range(0, len(params), chunksize):
                params_pre = params[:i]
                params_chunk = params[i : i + chunksize]
                params_post = params[i + chunksize :]
                jac_chunk = self._jacobian(params_pre, params_chunk, params_post)
                jac_img += target.jacobian_image(
                    parameters=identities[i : i + chunksize],
                    data=jac_chunk,
                )
        else:
            jac = self._jacobian(params[:0], params, params[0:0])
            jac_img += target.jacobian_image(parameters=identities, data=jac)

        return jac_img

    def gradient(
        self,
        params: Optional[ArrayLike] = None,
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
    ) -> ArrayLike:
        """Compute the gradient of the model likelihood with respect to its parameters."""

        jacobian_image = self.jacobian(params=params).flatten("data")

        data = self.target[self.window].flatten("data")
        mask = self.target[self.window].flatten("mask")
        model = self().flatten("data")
        if likelihood == "gaussian":
            weight = self.target[self.window].flatten("weight")
            gradient = backend.sum(
                jacobian_image * ((data - model) * weight * (~mask))[..., None], dim=0
            )
        elif likelihood == "poisson":
            gradient = backend.sum(
                jacobian_image * ((1 - data / model) * (~mask))[..., None],
                dim=0,
            )

        return gradient
