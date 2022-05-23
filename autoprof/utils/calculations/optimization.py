import numpy as np

def stocastic_k_delta_step(loss, params, scale, k = 3):

    delta = np.zeros(len(params[0]))

    for i in range(1,k+1):
        u = params[0] - params[i]
        norm_u = np.linalg.norm(u)
        delta = delta + u * (loss[0] - loss[i]) / norm_u**2

    delta /= k

    stoc = np.random.normal(scale = scale * np.linalg.norm(delta) / np.linalg.norm(scale))

    return delta + stoc
