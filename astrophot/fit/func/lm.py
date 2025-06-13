import torch
import numpy as np


def hessian(J, W):
    return J.T @ (W * J)


def gradient(J, W, R):
    return -J.T @ (W * R)


def step(L, grad, hess):
    I = torch.eye(len(grad), dtype=grad.dtype, device=grad.device)
    D = torch.ones_like(hess) - I

    h = torch.linalg.solve(
        hess * (I + D / (1 + L)) + L * I * (1 + torch.diag(hess)),
        grad,
    )

    return h


def step(x, data, model, weight, jacobian, ndf, chi2, L=1.0, Lup=9.0, Ldn=10.0):

    M0 = model(x)
    J = jacobian(x)
    R = data - M0
    grad = gradient(J, weight, R)
    hess = hessian(J, weight)

    best = {"h": torch.zeros_like(x), "chi2": chi2, "L": L}
    scary = {"h": None, "chi2": chi2, "L": L}

    improving = None
    for i in range(10):
        h = step(L, grad, hess)
        M1 = model(x + h)

        chi2 = torch.sum(weight * (data - M1) ** 2).item() / ndf

        # Handle nan chi2
        if not np.isfinite(chi2):
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if chi2 < scary["chi2"]:
            scary = {"h": h, "chi2": chi2, "L": L}
