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

# def local_derivative(loss, params):
#     assert len(params) == (len(params[0])+1)
#     assert len(params) == len(loss)
    
#     A = np.concatenate((params,loss.reshape(-1,1)), axis = 1)
#     n = np.linalg.solve(A,np.ones(len(A)))
#     return - n[:-1] / n[-1]

def local_derivative(x, y):
    if len(x[0]) == 1:
        return (x[1] - x[0]) / (y[0] - y[1])
    
    A = []
    b = []
    for i in range(1,len(x)):
        A.append(np.require(x[0] - x[i], dtype = float))
        b.append(y[0] - y[i])
    try:
        return np.linalg.solve(A,b)
    except np.linalg.LinAlgError:
        return np.zeros(len(b))
