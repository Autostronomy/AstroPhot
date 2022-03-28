from .substate_object import SubState
from autoprof.image import AP_Image

class Data_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.image = None
        self.mask = None
        self.psf = None

    def load_image(self, filename, pixelscale, **kwargs):
        pass
    def load_mask(self, filename, pixelscale, **kwargs):
        pass
    def load_psf(self, filename, pixelscale, **kwargs):
        pass
    def update_image(self, img):
        self.image = img
    def update_mask(self, img):
        self.mask = img
    def update_psf(self, img):
        self.psf = img
