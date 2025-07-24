from scipy.linalg import sqrtm
import numpy as np


def polar_decomposition(A):
    # Step 1: Compute symmetric positive-definite matrix P
    M = A.T @ A
    P = sqrtm(M)  # Principal square root of A^T A

    # Step 2: Compute rotation matrix R
    P_inv = np.linalg.inv(P)
    R = A @ P_inv
    return R, P
