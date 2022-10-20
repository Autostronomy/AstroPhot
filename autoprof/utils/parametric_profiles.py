import torch
import numpy as np

def sersic(R, n, Rs, I0):
    return I0 * torch.exp(-torch.pow(R/Rs,1/n))

def sersic_np(R, n, Rs, I0):
    return I0*np.exp(-(R/Rs)**(1/n))

def gaussian(R, I0, sigma):
    return I0 * torch.exp(-0.5*torch.pow(R/sigma,2))

def exponential(R, I0, Rd):
    return I0 * torch.exp(R / Rd)
