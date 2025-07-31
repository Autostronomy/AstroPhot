from functools import lru_cache
from scipy.special import binom
import numpy as np


@lru_cache(maxsize=1024)
def coefficients(n: int, m: int) -> list[tuple[int, float]]:
    C = []
    for k in range(int((n - abs(m)) / 2) + 1):
        C.append(
            (
                k,
                (-1) ** k * binom(n - k, k) * binom(n - 2 * k, (n - abs(m)) / 2 - k),
            )
        )
    return C


def zernike_n_m_list(n: int) -> list[tuple[int, int]]:
    nm = []
    for n_i in range(n + 1):
        for m_i in range(-n_i, n_i + 1, 2):
            nm.append((n_i, m_i))
    return nm


def zernike_n_m_modes(rho: np.ndarray, phi: np.ndarray, n: int, m: int) -> np.ndarray:
    Z = np.zeros_like(rho)
    for k, c in coefficients(n, m):
        R = rho ** (n - 2 * k)
        T = 1.0
        if m < 0:
            T = np.sin(abs(m) * phi)
        elif m > 0:
            T = np.cos(m * phi)

        Z = Z + c * R * T
    return Z * (rho <= 1).astype(np.float64)
