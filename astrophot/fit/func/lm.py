import numpy as np

from ...errors import OptimizeStopFail, OptimizeStopSuccess
from ...backend_obj import backend
from ... import config


def nll(D, M, W):
    """
    Negative log-likelihood for Gaussian noise.
    D: data
    M: model prediction
    W: weights
    """
    return 0.5 * backend.sum(W * (D - M) ** 2, dim=-1)


def nll_poisson(D, M):
    """
    Negative log-likelihood for Poisson noise.
    D: data
    M: model prediction
    """
    return backend.sum(M - D * backend.log(M + 1e-10), dim=-1)  # Adding small value to avoid log(0)


def gradient(J, W, D, M):
    return J.T @ (W * (D - M)).reshape(D.shape + (1,))


def gradient_poisson(J, D, M):
    return J.T @ (D / M - 1).reshape(D.shape + (1,))


def hessian(J, W):
    return J.T @ (W.reshape(W.shape + (1,)) * J)


def hessian_poisson(J, D, M):
    return J.T @ ((D / (M**2 + 1e-10)).reshape(D.shape + (1,)) * J)


def damp_hessian(hess, L):
    I = backend.eye(len(hess), dtype=config.DTYPE, device=config.DEVICE)
    D = backend.ones_like(hess) - I
    return hess * (I + D / (1 + L)) + L * I * backend.diag(hess)


def rho(nll0, nll1, h, hessD, grad):
    return (nll0 - nll1) / backend.abs(h.T @ hessD @ h - 2 * grad.T @ h)


def solve(hess, grad, L, check_finite=True, max_tries=50):
    """Solve the damped system for the step, adding to the diagonal until it can be solved.

    A singular matrix raises an error with torch but silently gives nan or inf
    with jax. With `check_finite` both count as a failure, so that the two
    backends take the same steps. It must be turned off under vmap, where the
    values can not be inspected.

    What is added to the diagonal starts at `L` and doubles on every try. It
    is never less than a small fraction of the diagonal, so that there is
    something to add when `L` is zero. `OptimizeStopFail` is raised when
    `max_tries` additions were not enough, or the system is not finite.
    """
    hessD = damp_hessian(hess, L)  # (N, N)
    if check_finite and not (
        bool(backend.all(backend.isfinite(hessD))) and bool(backend.all(backend.isfinite(grad)))
    ):
        raise OptimizeStopFail("The Hessian or gradient is not finite, there is no step to take")
    eye = backend.eye(len(hessD), dtype=config.DTYPE, device=config.DEVICE)
    # 1e-30 is for a hessian of zeros, where there is no scale to go by
    floor = 1e-8 * backend.sum(backend.abs(backend.diag(hess))) / len(hess) + 1e-30
    add = backend.maximum(L + backend.zeros_like(floor), floor)
    for _ in range(max_tries):
        try:
            h = backend.linalg.solve(hessD, grad)
            if not check_finite or bool(backend.all(backend.isfinite(h))):
                return hessD, h
        except backend.LinAlgErr:
            pass
        hessD = hessD + add * eye
        add = add * 2
    raise OptimizeStopFail(
        f"The Hessian is still singular after adding to its diagonal {max_tries} times"
    )


def solve_vmap(hess, grad, L):
    """`solve` for use under vmap, where the values of the step can not be inspected."""
    return solve(hess, grad, L, check_finite=False)


def invert_hessian(hess, batched=False):
    """Inverse of a hessian, or the pseudo-inverse if it is singular.

    Returns the matrix and whether the pseudo-inverse was needed. A singular
    matrix raises an error with torch but gives nan or inf with jax, both are
    treated the same way here. `batched` is for a stack of matrices.
    """
    inv = backend.vmap(backend.linalg.inv) if batched else backend.linalg.inv
    pinv = backend.vmap(backend.linalg.pinv) if batched else backend.linalg.pinv
    try:
        result = inv(hess)
        singular = not bool(backend.all(backend.isfinite(result))) and bool(
            backend.all(backend.isfinite(hess))
        )
    except Exception:
        singular = True
    if singular:
        return pinv(hess), True
    return result, False


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
    max_step_iter=10,
):
    L0 = L
    M0 = backend.detach(model(x))  # (M,)
    J = backend.detach(jacobian(x))  # (M, N)

    if likelihood == "gaussian":
        nll0 = nll(data, M0, weight).item()
        grad = gradient(J, weight, data, M0)  # (N, 1)
        hess = hessian(J, weight)  # (N, N)
    elif likelihood == "poisson":
        nll0 = nll_poisson(data, M0).item()
        grad = gradient_poisson(J, data, M0)  # (N, 1)
        hess = hessian_poisson(J, data, M0)  # (N, N)
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    del J

    if backend.allclose(grad, backend.zeros_like(grad)):
        raise OptimizeStopSuccess("Gradient is zero, optimization converged.")

    best = {"x": backend.zeros_like(x), "nll": nll0, "L": L}
    scary = {"x": None, "nll": np.inf, "L": None, "rho": np.inf}
    nostep = True
    improving = None
    for i in range(max_step_iter):
        hessD, h = solve(hess, grad, L)  # (N, N), (N, 1)
        M1 = model(x + h.squeeze(1))  # (M,)
        if likelihood == "gaussian":
            nll1 = nll(data, M1, weight).item()
        elif likelihood == "poisson":
            nll1 = nll_poisson(data, M1).item()

        # Handle nan chi2
        if not np.isfinite(nll1):
            L *= Lup
            if improving is True:
                break
            improving = False
            continue

        if backend.allclose(h, backend.zeros_like(h)) and L < 0.1:
            if i == 0:
                raise OptimizeStopSuccess("Step with zero length means optimization complete.")
            break

        # actual nll improvement vs expected from linearization
        _rho = rho(nll0, nll1, h, hessD, grad).item()

        if (nll1 < (nll0 + tolerance) and abs(_rho - 1) < abs(scary["rho"] - 1)) or (
            nll1 < scary["nll"] and _rho > -10
        ):
            scary = {"x": x + h.squeeze(1), "nll": nll1, "L": L0, "rho": _rho}

        # Avoid highly non-linear regions
        if _rho < 0.1 or _rho > 2:
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


def batch_lm_step(
    x,
    data,
    model,
    weight,
    mask,
    jacobian,
    L=1.0,
    Lup=9.0,
    Ldn=11.0,
    tolerance=1e-4,
    likelihood="gaussian",
    max_step_iter=3,
):
    L0 = L  # (D,)
    M0 = backend.detach(model(x))  # (D, M)
    J = backend.detach(jacobian(x))  # (D, M, N)
    data = data * mask
    M0 = M0 * mask
    weight = weight * mask
    J = J * mask.reshape(mask.shape + (1,))

    if likelihood == "gaussian":
        nll0 = nll(data, M0, weight)  # (D,)
        grad = backend.vmap(gradient)(J, weight, data, M0)  # (D, N, 1)
        hess = backend.vmap(hessian)(J, weight)  # (D, N, N)
    elif likelihood == "poisson":
        nll0 = nll_poisson(data, M0)  # (D,)
        grad = backend.vmap(gradient_poisson)(J, data, M0)  # (D, N, 1)
        hess = backend.vmap(hessian_poisson)(J, data, M0)  # (D, N, N)
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")

    del J

    new_x = backend.copy(x)
    new_nll = backend.copy(nll0)
    new_L = backend.copy(L)

    for _ in range(max_step_iter):
        hessD, h = backend.vmap(solve_vmap)(hess, grad, new_L)  # (D, N, N), (D, N, 1)
        M1 = model(x + h.squeeze(2))  # (D, M)
        if likelihood == "gaussian":
            nll1 = nll(data, M1, weight)  # (D,)
        elif likelihood == "poisson":
            nll1 = nll_poisson(data, M1)  # (D,)

        # actual nll improvement vs expected from linearization
        _rho = backend.vmap(rho)(nll0, nll1, h, hessD, grad).reshape(-1)  # (D,)

        good = backend.isfinite(nll1) & (nll1 < new_nll) & (_rho > 0.1) & (_rho < 2)
        new_x = backend.where(good[:, None], x + h.squeeze(2), new_x)
        new_nll = backend.where(good, nll1, new_nll)
        new_L = backend.where(good, new_L / Ldn, new_L * Lup)

    return {"x": new_x, "nll": backend.to_numpy(new_nll), "L": new_L}
