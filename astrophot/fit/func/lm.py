import torch
import numpy as np

from ...errors import OptimizeStopFail, OptimizeStopSuccess


def nll(D, M, W):
    """
    Negative log-likelihood for Gaussian noise.
    D: data
    M: model prediction
    W: weights
    """
    return 0.5 * torch.sum(W * (D - M) ** 2)


def nll_poisson(D, M):
    """
    Negative log-likelihood for Poisson noise.
    D: data
    M: model prediction
    """
    return torch.sum(M - D * torch.log(M + 1e-10))  # Adding small value to avoid log(0)


def gradient(J, W, D, M):
    return J.T @ (W * (D - M)).unsqueeze(1)


def gradient_poisson(J, D, M):
    return J.T @ (D / M - 1).unsqueeze(1)


def hessian(J, W):
    return J.T @ (W.unsqueeze(1) * J)


def hessian_poisson(J, D, M):
    return J.T @ ((D / (M**2 + 1e-10)).unsqueeze(1) * J)


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


def lm_step(
    x,
    data,
    model,
    weight,
    jacobian,
    L=1.0,
    Lup=9.0,
    Ldn=11.0,
    tolerance=1e-4,
    likelihood="gaussian",
):
    L0 = L
    M0 = model(x)  # (M,)
    J = jacobian(x)  # (M, N)

    if likelihood == "gaussian":
        nll0 = nll(data, M0, weight).item()  # torch.sum(weight * R**2).item() / ndf
        grad = gradient(J, weight, data, M0)  # (N, 1)
        hess = hessian(J, weight)  # (N, N)
    elif likelihood == "poisson":
        nll0 = nll_poisson(data, M0).item()
        grad = gradient_poisson(J, data, M0)  # (N, 1)
        hess = hessian_poisson(J, data, M0)  # (N, N)
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    if torch.allclose(grad, torch.zeros_like(grad)):
        raise OptimizeStopSuccess("Gradient is zero, optimization converged.")

    best = {"x": torch.zeros_like(x), "nll": nll0, "L": L}
    scary = {"x": None, "nll": np.inf, "L": None, "rho": np.inf}
    nostep = True
    improving = None
    for _ in range(10):
        hessD, h = solve(hess, grad, L)  # (N, N), (N, 1)
        M1 = model(x + h.squeeze(1))  # (M,)
        if likelihood == "gaussian":
            nll1 = nll(data, M1, weight).item()  # torch.sum(weight * (data - M1) ** 2).item() / ndf
        elif likelihood == "poisson":
            nll1 = nll_poisson(data, M1).item()

        # Handle nan chi2
        if not np.isfinite(nll1):
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if torch.allclose(h, torch.zeros_like(h)) and L < 0.1:
            raise OptimizeStopSuccess("Step with zero length means optimization complete.")

        # actual nll improvement vs expected from linearization
        rho = (nll0 - nll1) / torch.abs(h.T @ hessD @ h - 2 * grad.T @ h).item()

        if (nll1 < (nll0 + tolerance) and abs(rho - 1) < abs(scary["rho"] - 1)) or (
            nll1 < scary["nll"] and rho > -10
        ):
            scary = {"x": x + h.squeeze(1), "nll": nll1, "L": L0, "rho": rho}

        # Avoid highly non-linear regions
        if rho < 0.1 or rho > 2:
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if nll1 < best["nll"]:  # new best
            best = {"x": x + h.squeeze(1), "nll": nll1, "L": L}
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
        if (best["nll"] - nll0) / nll0 < -0.1:
            break

    if nostep:
        if scary["x"] is not None and (scary["nll"] - nll0) / nll0 < tolerance:
            return scary
        raise OptimizeStopFail("Could not find step to improve chi^2")

    return best
