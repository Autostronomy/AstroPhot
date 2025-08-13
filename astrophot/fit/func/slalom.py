import numpy as np

from ...errors import OptimizeStopFail, OptimizeStopSuccess
from ...backend_obj import backend


def slalom_step(f, g, x0, m, S, N=10, up=1.3, down=0.5):
    l = [f(x0).item()]
    d = [0.0]
    grad = g(x0)
    if backend.allclose(grad, backend.zeros_like(grad)):
        raise OptimizeStopSuccess("success: Gradient is zero, optimization converged.")

    D = grad + m
    D = D / backend.linalg.norm(D)
    seeking = False
    for _ in range(N):
        l.append(f(x0 - S * D).item())
        d.append(S)

        # Check if the last value is finite
        if not np.isfinite(l[-1]):
            l.pop()
            d.pop()
            S *= down
            continue

        if seeking and np.argmin(l) == len(l) - 1:
            # If we are seeking a minimum and the last value is the minimum, we can stop
            break

        if len(l) < 3:
            # Seek better step size based on loss improvement
            if l[-1] < l[-2]:
                S *= up
            else:
                S *= down
        else:
            O = np.polyfit(d[-3:], l[-3:], 2)
            if O[0] > 0:
                S = -O[1] / (2 * O[0])
                seeking = True
            else:
                S *= down
                seeking = False

    if np.argmin(l) == 0:
        raise OptimizeStopFail("fail: cannot find step to improve.")
    return d[np.argmin(l)], l[np.argmin(l)], grad
