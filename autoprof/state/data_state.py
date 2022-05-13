from .substate_object import SubState
from autoprof.image import AP_Image
from autoprof.utils.image_operations import load_fits

class Data_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image = None
        self.sigma = None
        self.mask = None
        self.psf = None

    
    def load(self, filename, pixelscale, **kwargs):
        if filename.endswith('.fits'):
            hduelement = kwargs['index'] if 'index' in kwargs else 0
            img = AP_Image(load_fits(filename, hduelement), pixelscale = pixelscale, **kwargs)
        else:
            raise ValueError(f'Unrecognized filetype for {filename}. Try converting to FITS image type.')
        return img
        
    def update_image(self, img, **kwargs):
        if isinstance(img, AP_Image):
            self.image = img
        elif isinstance(img, str):
            self.image = self.load(img, **kwargs)
    def update_sigma(self, img, **kwargs):
        if isinstance(img, AP_Image):
            self.sigma = img
        elif isinstance(img, str):
            self.sigma = self.load(img, **kwargs)
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
    def update_psf(self, img):
        if isinstance(img, AP_Image):
            self.psf = img
        elif isinstance(img, str):
            self.psf = self.load(img, **kwargs)
