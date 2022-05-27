from .substate_object import SubState
from autoprof.image import AP_Image, PSF_Image, Model_Image
from astropy.io import fits
import numpy as np

class Data_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target = None
        self.variance = None
        self.mask = None
        self.psf = None
        self.loss_image = None
        self.model_image = None
    
    def load(self, filename, pixelscale, **kwargs):
        if "image_type" in kwargs:
            image_type = kwargs["image_type"]
        else:
            image_type = AP_Image
            
        if filename.endswith('.fits'):
            hdulelement = kwargs['index'] if 'index' in kwargs else 0
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
            self.variance = img
        elif isinstance(img, str):
            self.variance = self.load(img, **kwargs)
            
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

    def initialize_model_image(self):
        window_origin = None
        window_shape = None
        for model in self.state.models:
            if model.locked:
                continue
            if window_origin is None:
                window_origin = list(model.model_image.origin)
                window_shape = list(model.window)
                continue
            window_origin[0] = min(window_origin[0], model.model_image.origin[0])
            window_origin[1] = min(window_origin[1], model.model_image.origin[1])
            window_shape = [
                slice(min(window_shape[0].start, model.window[0].start),
                      max(window_shape[0].stop, model.window[0].stop)),
                slice(min(window_shape[1].start, model.window[1].start),
                      max(window_shape[1].stop, model.window[1].stop)),
            ]
        self.model_image = Model_Image(
            np.zeros((window_shape[0].stop - window_shape[0].start,
                      window_shape[1].stop - window_shape[1].start)),
            pixelscale=self.target.pixelscale,
            origin=np.array(window_origin),
        )
        
        
