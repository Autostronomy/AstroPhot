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

def Angle_COM_PA(flux, X=None, Y=None):
    """Performs a center of angular mass calculation by using the flux as
    weights to compute a position angle which accounts for the general
    "direction" of the light. This PA is computed mod pi since these
    are 180 degree rotation symmetric.

    Args:
      flux: the weight values for each element (by assumption, pixel fluxes) in a 2D array
      X: x coordinate of the flux points. Assumed centered pixel indices if not given
      Y: y coordinate of the flux points. Assumed centered pixel indices if not given

    """
    if X is None:
        S = flux.shape
        X, Y = np.meshgrid(np.arange(S[1]) - S[1]/2, np.arange(S[0]) - S[0]/2, indexing = "xy")
        
    theta = np.arctan2(Y, X)

    ang_com_cos = np.sum(flux * np.cos(2*theta)) / np.sum(flux)
    ang_com_sin = np.sum(flux * np.sin(2*theta)) / np.sum(flux)

    return np.arctan2(ang_com_sin, ang_com_cos)/2 % np.pi
