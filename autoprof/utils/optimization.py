import numpy as np

def k_delta_step(loss, params, k = 3, reference = 0):

    delta = np.zeros(len(params[reference]), dtype = float)
    for i in range(k+1):
        if i == reference: continue
        u = params[reference] - params[i]
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            k -= 1
            continue
        delta = delta + u * (loss[reference] - loss[i]) / norm_u**2
    if k <= 0:
        return np.zeros(len(delta))
    return delta / k

