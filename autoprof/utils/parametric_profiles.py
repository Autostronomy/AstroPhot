import torch
import numpy as np

def sersic(R, n, Rs, I0):
    return I0 * torch.exp(-torch.pow((R+1e-6)/Rs, 1/n)) # epsilon added for numerical stability of gradient
def sersic_np(R, n, Rs, I0):
    if np.any(np.array([n, Rs, I0]) <= 0):
        return np.ones(len(R))*1e6
    return I0*(np.exp(-(R/Rs)**(1/n)) + 1e-6)

def gaussian_torch(R, sigma, I0):
    return (I0 / torch.sqrt(2 * np.pi * sigma**2)) * torch.exp(-0.5*torch.pow(R/sigma,2))
def gaussian_np(R, sigma, I0):
    return (I0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5*((R/sigma)**2))

def exponential(R, I0, Rd):
    return I0 * torch.exp(R / Rd)
