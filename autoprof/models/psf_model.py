from .star_model_object import Star_Model

__all__ = ["PSF_Star"]

class PSF_Star(Star_Model):
    """Star model which uses an image of the PSF as it's representation
    for stars. Using Lanczos interpolation it will shift the PSF
    within a pixel to accurately represent the center location of a
    point source.

    """
    
    model_type = f"psf {Star_Model.model_type}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psf_model = Model_image(data = self.target.psf, pixelscale = self.target.pixelscale)

    def evaluate_model(self, image):# fixme this definitely has bugs
        shift = ((self["center"].value/self.psf_model.pixelscale) % 1.)
        psf = shift_Lanczos(self.psf_model.data, shift[0], shift[1], min(min(self.psf_model.data.shape), 10))
        psf = psf * self["flux"] / torch.sum(psf)
        res = Model_Image(data = torch.zeros(image.data.shape), pixelscale = image.pixelscale)
        icenter = coord_to_index(self["center"].value[0], self["center"].value[1], image)
        psf_edge = (self.psf_model.data.shape - 1) / 2
        psf_model = Model_Image(data = psf, shape = psf.shape*self.psf_model.pixelscale, origin = torch.floor(self["center"].value/self.psf_model.pixelscale) - psf_edge)
        res += psf_model
        return res.data
