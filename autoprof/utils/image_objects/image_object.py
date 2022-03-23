
class Single_Image(object):

    def __init__(self, pixelscale, filename = None, img = None, zeropoint = None, hduelement = 0, **kwargs):

        assert filename or img
        assert isinstance(pixelscale, float)

        self.pixelscale = pixelscale
        if filename:
            self.location = filename
            self.hduelement = hduelement
            self.img = read(filename, hduelement)
        elif img:
            self.imgs = img
        self.size = self.img.size

        self.zeropoint = zeropoint

class Image(object):

    interpolation_method = 'lanczos'
    
    def __init__(self, pixelscale, **kwargs):

        self.image_variants = []
        if len(args) or len(kwargs):
            self.image_variants.append(Single_Image(pixelscale, **kwargs))
            self.current_image = 0

        
