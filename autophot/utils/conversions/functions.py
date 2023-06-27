import numpy as np
import torch
from scipy.special import gamma
from torch.special import gammaln


def sersic_n_to_b(n):
    return (
        2 * n
        - 1 / 3
        + 4 / (405 * n)
        + 46 / (25515 * n ** 2)
        + 131 / (1148175 * n ** 3)
        - 2194697 / (30690717750 * n ** 4)
    )


def sersic_I0_to_flux_np(I0, n, R, q):
    return 2 * np.pi * I0 * q * n * R ** 2 * gamma(2 * n)


def sersic_flux_to_I0_np(flux, n, R, q):
    return flux / (2 * np.pi * q * n * R ** 2 * gamma(2 * n))


def sersic_Ie_to_flux_np(Ie, n, R, q):
    bn = sersic_n_to_b(n)
    return (
        2 * np.pi * Ie * R ** 2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n)
    )


def sersic_flux_to_Ie_np(flux, n, R, q):
    bn = sersic_n_to_b(n)
    return flux / (
        2 * np.pi * R ** 2 * q * n * (np.exp(bn) * bn ** (-2 * n)) * gamma(2 * n)
    )


def sersic_inv_np(I, n, Re, Ie):
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * np.log(I / Ie)) ** (n))


def sersic_I0_to_flux_torch(I0, n, R, q):
    return 2 * np.pi * I0 * q * n * R ** 2 * torch.exp(gammaln(2 * n))


def sersic_flux_to_I0_torch(flux, n, R, q):
    return flux / (2 * np.pi * q * n * R ** 2 * torch.exp(gammaln(2 * n)))


def sersic_Ie_to_flux_torch(Ie, n, R, q):
    bn = sersic_n_to_b(n)
    return (
        2
        * np.pi
        * Ie
        * R ** 2
        * q
        * n
        * (torch.exp(bn) * bn ** (-2 * n))
        * torch.exp(gammaln(2 * n))
    )

def sersic_flux_to_Ie_torch(flux, n, R, q):
    bn = sersic_n_to_b(n)
    return flux / (
        2
        * np.pi
        * R ** 2
        * q
        * n
        * (torch.exp(bn) * bn ** (-2 * n))
        * torch.exp(gammaln(2 * n))
    )


def sersic_inv_torch(I, n, Re, Ie):
    bn = sersic_n_to_b(n)
    return Re * ((1 - (1 / bn) * torch.log(I / Ie)) ** (n))

def moffat_I0_to_flux(I0, n, rd, q):
    return I0 * np.pi * rd**2 * q / (n - 1)

def general_uncertainty_prop(
        param_tuple, #tuple of parameter values
        param_err_tuple, # tuple of parameter uncertainties
        forward # forward function through which to get uncertainty
):
    # Make a new set of parameters which track uncertainty
    new_params = []
    for p in param_tuple:
        newp = p.detach()
        newp.requires_grad = True
        new_params.append(newp)
    # propogate forward and compute derivatives
    f = forward(*new_params)
    f.backward()
    # Add all the error contributions in quadrature
    x = torch.zeros_like(f)
    for i in range(len(new_params)):
        x = x + (new_params[i].grad * param_err_tuple[i])**2
    return x.sqrt()
