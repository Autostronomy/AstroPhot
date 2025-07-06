from functools import lru_cache

import torch
from scipy.special import binom

from ..utils.decorators import ignore_numpy_warnings
from .psf_model_object import PSFModel
from ..errors import SpecificationConflict
from ..param import forward

__all__ = ("ZernikePSF",)


class ZernikePSF(PSFModel):

    _model_type = "zernike"
    _parameter_specs = {"Anm": {"units": "flux/arcsec^2"}}
    usable = True

    def __init__(self, *args, order_n=2, r_scale=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.order_n = int(order_n)
        self.r_scale = r_scale
        self.nm_list = self.iter_nm(self.order_n)

    @torch.no_grad()
    @ignore_numpy_warnings
    def initialize(self):
        super().initialize()

        # List the coefficients to use
        self.nm_list = self.iter_nm(self.order_n)
        # Set the scale radius for the Zernike area
        if self.r_scale is None:
            self.r_scale = max(self.window.shape) / 2

        # Check if user has already set the coefficients
        if self.Anm.initialized:
            if len(self.nm_list) != len(self.Anm.value):
                raise SpecificationConflict(
                    f"nm_list length ({len(self.nm_list)}) must match coefficients ({len(self.Anm.value)})"
                )
            return

        # Set the default coefficients to zeros
        self.Anm.dynamic_value = torch.zeros(len(self.nm_list))
        self.Anm.uncertainty = self.default_uncertainty * torch.ones_like(self.Anm.value)
        if self.nm_list[0] == (0, 0):
            self.Anm.value[0] = torch.median(self.target[self.window].data) / self.target.pixel_area

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
                    (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - abs(m)) / 2 - k),
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

    @forward
    def brightness(self, x, y, Anm):
        x, y = self.transform_coordinates(x, y)

        phi = self.angular_metric(x, y)

        r = self.radius_metric(x, y)
        r = r / self.r_scale

        G = torch.zeros_like(x)

        i = 0
        for n, m in self.nm_list:
            G += Anm[i] * self.Z_n_m(r, phi, n, m)
            i += 1

        G[r > 1] = 0.0

        return G
