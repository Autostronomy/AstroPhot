import numpy as np

def Angle_TwoAngles_sin(a1, a2):
    """
    Compute the angle between two vectors at angles a1 and a2
    """

    return np.arcsin(np.sin(a1 - a2))


def Angle_TwoAngles_cos(a1, a2):
    """
    Compute the angle between two vectors at angles a1 and a2
    """

    return np.arccos(np.cos(a1 - a2))


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
