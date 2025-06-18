import torch
import numpy as np

from ...errors import OptimizeStop


def hessian(J, W):
    return J.T @ (W * J)


def gradient(J, W, R):
    return -J.T @ (W * R)


def damp_hessian(hess, L):
    I = torch.eye(len(hess), dtype=hess.dtype, device=hess.device)
    D = torch.ones_like(hess) - I
    return hess * (I + D / (1 + L)) + L * I * (1 + torch.diag(hess))


def step(x, data, model, weight, jacobian, ndf, chi2, L=1.0, Lup=9.0, Ldn=10.0):

    chi20 = chi2
    M0 = model(x)
    J = jacobian(x)
    R = data - M0
    grad = gradient(J, weight, R)
    hess = hessian(J, weight)

    best = {"h": torch.zeros_like(x), "chi2": chi20, "L": L}
    scary = {"h": None, "chi2": chi20, "L": L}

    nostep = True
    improving = None
    for _ in range(10):
        hessD = damp_hessian(hess, L)
        h = torch.linalg.solve(hessD, grad)
        M1 = model(x + h)

        chi21 = torch.sum(weight * (data - M1) ** 2).item() / ndf

        # Handle nan chi2
        if not np.isfinite(chi21):
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if chi21 < scary["chi2"]:
            scary = {"h": h, "chi2": chi21, "L": L}

        # actual chi2 improvement vs expected from linearization
        rho = (chi20 - chi21) / torch.abs(h.T @ hessD @ h - 2 * grad @ h).item()

        # Avoid highly non-linear regions
        if rho < 0.1 or rho > 10:
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if chi21 < best["chi2"]:  # new best
            best = {"h": h, "chi2": chi21, "L": L}
            nostep = False
            L /= Ldn
            if L < 1e-8 or improving is False:
                break
            improving = True
        elif improving is True:
            break
        else:  # not improving and bad chi2, damp more
            L *= Lup
            if L >= 1e9:
                break
            improving = False

        if (best["chi2"] - chi20) / chi20 < -0.1:
            # If we are improving chi2 by more than 10% then we can stop
            break

    if nostep:
        if scary["h"] is not None:
            return scary
        raise OptimizeStop("Could not find step to improve chi^2")

    return best
