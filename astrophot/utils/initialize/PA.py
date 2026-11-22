import numpy as np


def polar_decomposition(A):
    # Step 1: Compute symmetric positive-definite matrix P
    from scipy.linalg import sqrtm  # here to keep `import astrophot` fast

    M = A.T @ A
    P = sqrtm(M)  # Principal square root of A^T A

    # Step 2: Compute rotation matrix R
    P_inv = np.linalg.inv(P)
    R = A @ P_inv
    return R, P


def R(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
