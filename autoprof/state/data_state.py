from .substate_object import SubState
from autoprof.image import AP_Image, PSF_Image, Model_Image
from astropy.io import fits
import numpy as np
from copy import deepcopy

class Data_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target = None
        self.mask = None
        self.psf = None
        self.variance_image = None
        self.loss_image = None
        self.residual_image = None
        self.model_image = None
    
    def load(self, filename, pixelscale, **kwargs):
        if "image_type" in kwargs:
            image_type = kwargs["image_type"]
        else:
            image_type = AP_Image
            
        if filename.endswith('.fits'):
            hdulelement = kwargs.get('index', 0)
            hdul = fits.open(filename)
            img = image_type(hdul[hdulelement].data, pixelscale = pixelscale, **kwargs)
        elif filename.endswith('.npy'):
            img = image_type(np.require(np.load(filename),dtype=float), pixelscale = pixelscale, **kwargs)
        else:
            raise ValueError(f'Unrecognized filetype for {filename}. Try converting to FITS image type.')
        return img
        
    def update_target(self, img, **kwargs):
        if isinstance(img, AP_Image):
            self.target = img
        elif isinstance(img, str):
            self.target = self.load(img, **kwargs)
            
    def update_variance(self, img, **kwargs):
        if isinstance(img, AP_Image):
            self.variance_image = img
        elif isinstance(img, str):
            self.variance_image = self.load(img, **kwargs)
        elif isinstance(img, np.ndarray):
            self.variance_image = AP_Image(img, **kwargs)
            
    def update_mask(self, img, mode = 'or', **kwargs):
        if isinstance(img, AP_Image):
            newmask = img
        elif isinstance(img, str):
            newmask = self.load(img, **kwargs)

        if self.mask is None:
            self.mask = newmask
            return
        if mode == 'or':
            self.mask = np.logical_or(self.mask, newmask)
        elif mode == 'and':
            self.mask = np.logical_and(self.mask, newmask)
        else:
            raise ValueError(f'unrecognized logical operation {mode}, must be one of: or, and')
        
    def update_psf(self, img, **kwargs):
        if isinstance(img, PSF_Image):
            self.psf = img
        elif isinstance(img, str):
            self.psf = self.load(img, image_type = PSF_Image, **kwargs)
        elif isinstance(img, np.ndarray):
            self.psf = PSF_Image(img, **kwargs)

    def initialize_model_image(self, full_target = False, include_locked = False):

        if full_target:
            self.model_image = Model_Image(
                np.zeros(np.round(self.target.shape / self.target.pixelscale).astype(int)),
                pixelscale = self.target.pixelscale,
                window = self.target.window,
            )
            return
        new_window = None
        for model in self.state.models:
            if model.locked and not include_locked:
                continue
            if new_window is None:
                new_window = deepcopy(model.window)
            else:
                new_window += model.window
                
        self.model_image = Model_Image(
            np.zeros(np.round(np.array(new_window.shape) / self.target.pixelscale).astype(int)),
            pixelscale=self.target.pixelscale,
            origin=new_window.origin,
        )
        
        
