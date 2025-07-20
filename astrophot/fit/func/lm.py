import torch
import numpy as np

from ...errors import OptimizeStopFail, OptimizeStopSuccess


def hessian(J, W):
    return J.T @ (W.unsqueeze(1) * J)


def gradient(J, W, R):
    return J.T @ (W * R).unsqueeze(1)


def damp_hessian(hess, L):
    I = torch.eye(len(hess), dtype=hess.dtype, device=hess.device)
    D = torch.ones_like(hess) - I
    return hess * (I + D / (1 + L)) + L * I * torch.diag(hess)


def solve(hess, grad, L):
    hessD = damp_hessian(hess, L)  # (N, N)
    while True:
        try:
            h = torch.linalg.solve(hessD, grad)
            break
        except torch._C._LinAlgError:
            hessD = hessD + L * torch.eye(len(hessD), dtype=hessD.dtype, device=hessD.device)
            L = L * 2
    return hessD, h


def lm_step(x, data, model, weight, jacobian, ndf, chi2, L=1.0, Lup=9.0, Ldn=11.0):
    chi20 = chi2
    M0 = model(x)  # (M,)
    J = jacobian(x)  # (M, N)
    R = data - M0  # (M,)
    grad = gradient(J, weight, R)  # (N, 1)
    hess = hessian(J, weight)  # (N, N)
    if torch.allclose(grad, torch.zeros_like(grad)):
        raise OptimizeStopSuccess("Gradient is zero, optimization converged.")

    best = {"x": torch.zeros_like(x), "chi2": chi20, "L": L}
    scary = {"x": None, "chi2": chi20, "L": L}

    nostep = True
    improving = None
    for _ in range(10):
        hessD, h = solve(hess, grad, L)  # (N, N), (N, 1)
        M1 = model(x + h.squeeze(1))  # (M,)
        chi21 = torch.sum(weight * (data - M1) ** 2).item() / ndf

        # Handle nan chi2
        if not np.isfinite(chi21):
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if chi21 < scary["chi2"]:
            scary = {"x": x + h.squeeze(1), "chi2": chi21, "L": L}

        # if torch.allclose(h, torch.zeros_like(h)):
        #     raise OptimizeStopSuccess("Step with zero length means optimization complete.")

        # actual chi2 improvement vs expected from linearization
        rho = (chi20 - chi21) * ndf / torch.abs(h.T @ hessD @ h - 2 * grad.T @ h).item()
        # Avoid highly non-linear regions
        if rho < 0.1 or rho > 10:
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if chi21 < best["chi2"]:  # new best
            best = {"x": x + h.squeeze(1), "chi2": chi21, "L": L}
            nostep = False
            L /= Ldn
            if L < 1e-8 or improving is False:
                break
            improving = True
        elif improving is True:  # were improving, now not improving
            break
        else:  # not improving and bad chi2, damp more
            L *= Lup
            if L >= 1e9:
                break
            improving = False

        # If we are improving chi2 by more than 10% then we can stop
        if (best["chi2"] - chi20) / chi20 < -0.1:
            break

    if nostep:
        if scary["x"] is not None:
            return scary
        raise OptimizeStopFail("Could not find step to improve chi^2")

    return best
