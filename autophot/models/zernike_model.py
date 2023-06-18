import torch
import numpy as np
from scipy.special import binom

from .star_model_object import Star_Model



class Zernike_Star(Star_Model):

    model_type = f"zernike {Star_Model.model_type}"
    parameter_specs = {
        "a0": {"units": "flux/arcsec^2"},
        "anm": {"units": "flux/arcsec^2"},
        "bnm": {"units": "flux/arcsec^2"},
    }
    _parameter_order = Component_Model._parameter_order + ("q", "PA")
    useable = False
    
    
    def __init__(self, name, *args, order_n = 2, r_scale = None, **kwargs):

        self.order_n = int(order_n)
        self.r_scale = r_scale

    @lru_cache(maxsize = 32)
    def iter_nm(self, n):
        nm = []
        for n_i in range(n+1):
            for m_i in range(0 if n_i % 2 == 0 else 1, n_i+1, 2):
                nm.append(( n_i, m_i ))
        return nm
        
    def coefficients(self, n, m):
        C = []
        for k in range((n - m)/2):
            C.append((k, (-1)**k * binom(n - k, k) * binom(n - 2*k, (n-m)/2 - k)))
        return C

    def Z_n_m(self, rho, phi, n, m):
        Z = torch.zeros_like(rho)
        for k, c in self.coefficients(n, m):
            if m == 0:
                Z += c * rho**(n - 2*k)
            elif m < 0:
                Z += c * rho**(n - 2*k) * torch.sin(m * phi)
            else:
                Z += c * rho**(n - 2*k) * torch.cos(m * phi)
        return Z
            
                
    def evaluate_model(self, X = None, Y = None, image = None, parameters = None):
        if X is None:
            Coords = image.get_coordinate_meshgrid()
            X, Y = Coords - parameters["center"].value[...,None, None]

        phi = self.angular_metric(X, Y, image, parameters)

        r = self.radius_metric(X, Y, image, parameters)
        r = r / self.r_scale

        G = torch.zeros_like(X) + parameters["a0"].value
        
        for (n_i, m_i), c_i in zip(self.iter_nm(self.order_n), self.coefficients(self.order_n)):
            if n_i == 0:
                continue
            if m_i >= 0:
                G += Z_n_m(r, phi, n_i, m_i) * parameters["anm"].value[]
            else:
                G += Z_n_m(r, phi, n_i, m_i) * parameters["bnm"].value[]
                
        G[r > 1] = 0.

        return G
