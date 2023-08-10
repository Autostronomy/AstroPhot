from functools import lru_cache

import torch
import numpy as np
from scipy.special import binom

from ..utils.decorators import ignore_numpy_warnings, default_internal
from ._shared_methods import select_target
from .star_model_object import Star_Model
from .. import AP_config

__all__ = ("Zernike_Star",)


class Zernike_Star(Star_Model):

    model_type = f"zernike {Star_Model.model_type}"
    parameter_specs = {
        "Anm": {"units": "flux/arcsec^2"},
    }
    _parameter_order = Star_Model._parameter_order + ("Anm",)
    useable = True

    def __init__(self, name, *args, order_n=2, r_scale=None, **kwargs):
        super().__init__(name, *args, **kwargs)

        self.order_n = int(order_n)
        self.r_scale = r_scale
        self.nm_list = self.iter_nm(self.order_n)

    @torch.no_grad()
    @ignore_numpy_warnings
    @select_target
    @default_internal
    def initialize(self, target=None, parameters=None, **kwargs):
        super().initialize(target=target, parameters=parameters)

        # List the coefficients to use
        self.nm_list = self.iter_nm(self.order_n)
        # Set the scale radius for the Zernike area
        if self.r_scale is None:
            self.r_scale = torch.max(self.window.shape) / 2

        # Check if user has already set the coefficients
        if parameters["Anm"].value is not None:
            assert len(self.nm_list) == len(
                parameters["Anm"].value
            ), "nm_list must match coefficients (Anm)"
            return

        # Set the default coefficients to zeros
        parameters["Anm"].set_value(
            torch.zeros(len(self.nm_list)), override_locked=True
        )

        # Set the zero order zernike polynomial to the average in the image
        if self.nm_list[0] == (0, 0):
            parameters["Anm"].value[0] = (
                torch.median(target[self.window].data) / target.pixel_area
            )

    def iter_nm(self, n):
        nm = []
        for n_i in range(n + 1):
            for m_i in range(-n_i, n_i + 1, 2):
                nm.append((n_i, m_i))
        return nm

    @staticmethod
    @lru_cache(maxsize=1024)
    def coefficients(n, m):
        C = []
        for k in range(int((n - abs(m)) / 2) + 1):
            C.append(
                (
                    k,
                    (-1) ** k
                    * binom(n - k, k)
                    * binom(n - 2 * k, (n - abs(m)) / 2 - k),
                )
            )
        return C

    def Z_n_m(self, rho, phi, n, m, efficient=True):
        Z = torch.zeros_like(rho)
        if efficient:
            T_cache = {0: None}
            R_cache = {}
        for k, c in self.coefficients(n, m):
            if efficient:
                if (n - 2 * k) not in R_cache:
                    R_cache[n - 2 * k] = rho ** (n - 2 * k)
                R = R_cache[n - 2 * k]
                if m not in T_cache:
                    if m < 0:
                        T_cache[m] = torch.sin(abs(m) * phi)
                    elif m > 0:
                        T_cache[m] = torch.cos(m * phi)
                T = T_cache[m]
            else:
                R = rho ** (n - 2 * k)
                if m < 0:
                    T = torch.sin(abs(m) * phi)
                elif m > 0:
                    T = torch.cos(m * phi)

            if m == 0:
                Z += c * R
            elif m < 0:
                Z += c * R * T
            else:
                Z += c * R * T
        return Z

    def evaluate_model(self, X=None, Y=None, image=None, parameters=None):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[..., None, None]

        phi = self.angular_metric(X, Y, image, parameters)

        r = self.radius_metric(X, Y, image, parameters)
        r = r / self.r_scale

        G = torch.zeros_like(X)

        i = 0
        A = image.pixel_area * parameters["Anm"].value
        for n, m in self.nm_list:
            G += A[i] * self.Z_n_m(r, phi, n, m)
            i += 1

        G[r > 1] = 0.0

        return G
