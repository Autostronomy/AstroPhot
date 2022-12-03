import torch
import matplotlib.pyplot as plt
import numpy as np

def fft_convolve_torch(img, psf, psf_fft = False, img_prepadded = False):
    
    # Ensure everything is tensor
    img = torch.as_tensor(img)
    psf = torch.as_tensor(psf)
    
    if img_prepadded:
        s = list(int(2**(np.ceil(np.log2(d)))) for d in img.size())
    else:
        s = list(int(2**(np.ceil(np.log2(d + (p+1)/2)))) for d, p in zip(img.size(), psf.size()))
        
    img_f = torch.fft.rfft2(img, s = s)
    if not psf_fft:
        psf_f = torch.fft.rfft2(psf, s = s)
    else:
        psf_f = psf

    conv_f = img_f * psf_f

    conv = torch.fft.irfft2(conv_f)
    return torch.roll(conv, shifts = (-int((psf.size()[0]-1)/2),-int((psf.size()[1]-1)/2)), dims = (0,1))[:img.size()[0],:img.size()[1]]


if __name__ == "__main__":
    my_image = np.zeros((100,100))
    my_image[3,7] = 1
    my_image[20,65:82] = 1
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
    
