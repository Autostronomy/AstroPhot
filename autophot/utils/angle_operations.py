import numpy as np
from scipy.stats import iqr


def Angle_Average(a):
    """
    Compute the average for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j * np.sin(a)
    return np.angle(np.mean(i))


def Angle_Median(a):
    """
    Compute the median for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.median(np.cos(a)) + 1j * np.median(np.sin(a))
    return np.angle(i)


def Angle_Scatter(a):
    """
    Compute the scatter for a list of angles, which may wrap around a cyclic boundary.

    a: list of angles in the range [0,2pi]
    """
    i = np.cos(a) + 1j * np.sin(a)
    return iqr(np.angle(1j * i / np.mean(i)), rng=[16, 84])
