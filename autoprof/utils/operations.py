import torch
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve_fft, convolve


def fft_convolve_torch(img, psf, psf_fft = False, img_prepadded = False):
    # Ensure everything is tensor
    img = torch.as_tensor(img)
    psf = torch.as_tensor(psf)
    
    if img_prepadded:
        s = img.size() 
    else:
        s = list(int(d + (p+1)/2) for d, p in zip(img.size(), psf.size()))
        
    img_f = torch.fft.rfft2(img, s = s)
    if not psf_fft:
        psf_f = torch.fft.rfft2(psf, s = s)
    else:
        psf_f = psf

    conv_f = img_f * psf_f
    conv = torch.fft.irfft2(conv_f, s = s)
    
    return torch.roll(conv, shifts = (-int((psf.size()[0]-1)/2),-int((psf.size()[1]-1)/2)), dims = (0,1))[:img.size()[0],:img.size()[1]]

def fft_convolve_multi_torch(img, kernels, kernel_fft = False, img_prepadded = False, dtype = None, device = None):
    # Ensure everything is tensor
    img = torch.as_tensor(img, dtype = dtype, device = device)
    for k in range(len(kernels)):
        kernels[k] = torch.as_tensor(kernels[k], dtype = dtype, device = device)
    
    if img_prepadded:
        s = img.size() 
    else:
        s = list(int(d + (p+1)/2) for d, p in zip(img.size(), kernels[0].size()))
        
    img_f = torch.fft.rfft2(img, s = s)
    if not kernel_fft:
        kernels_f = list(torch.fft.rfft2(kernel, s = s) for kernel in kernels)
    else:
        psf_f = psf

    conv_f = img_f
    for kernel_f in kernels_f:
        conv_f *= kernel_f
    conv = torch.fft.irfft2(conv_f, s = s)
    return torch.roll(conv, shifts = (-int((sum(kernel.size()[0] for kernel in kernels)-1)/2),-int((sum(kernel.size()[1] for kernel in kernels)-1)/2)), dims = (0,1))[:img.size()[0],:img.size()[1]]


if __name__ == "__main__":
    my_image = np.zeros((100,100))
    my_image[3,7] = 1
    my_image[20,62:82] = np.logspace(-20,0,20)
    my_image[73:78,31] = 1
    my_image[87:90,83:85] = 1

    XX, YY = np.meshgrid(np.linspace(-1,1,21), np.linspace(-1,1,21))
    ZZ = np.exp(-(XX**2 + YY**2)/ (0.1**2))
    my_psf = ZZ / np.sum(ZZ)
    print(np.max(my_psf))
    plt.imshow(my_image, origin = "lower")
    plt.title("image")
    plt.show()
    plt.imshow(np.log10(my_psf), origin = "lower")
    plt.title("psf")
    plt.show()
    conv = fft_convolve_torch(my_image, my_psf, img_prepadded = False)
    print(conv.shape)
    print(torch.max(conv))
    plt.imshow(np.log10(conv.detach().numpy()), origin = "lower")
    plt.title("convolved")
    plt.show()
    XX, YY = np.meshgrid(np.linspace(-1,1,11), np.linspace(-1,1,11))
    ZZ = np.exp(-(XX**2 + YY**2)/ (0.2**2))
    my_psf2 = ZZ / np.sum(ZZ)

    conv2 = fft_convolve_multi_torch(my_image, [my_psf, my_psf2], img_prepadded = False)
    print(conv2.shape)
    print(torch.max(conv2))
    plt.imshow(np.log10(conv2.detach().numpy()), origin = "lower")
    plt.title("convolved2")
    plt.show()

    
    
