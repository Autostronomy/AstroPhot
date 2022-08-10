import numpy as np

class AP_Window(object):

    def __init__(self, origin, shape):

        self.shape = np.array(shape)
        self.origin = np.array(origin)
        self.center = self.origin + self.shape/2

    def get_indices(self, obj):
        """
        Return an index slicing tuple for obj corresponding to this window
        """
        return (
            slice(max(0,int(round((self.origin[0] - obj.window.origin[0])/obj.pixelscale))),
                  min(int(round(obj.window.shape[0]/obj.pixelscale)), int(round((self.origin[0] + self.shape[0] - obj.window.origin[0])/obj.pixelscale)))),
            slice(max(0,int(round((self.origin[1] - obj.window.origin[1])/obj.pixelscale))),
                  min(int(round(obj.window.shape[1]/obj.pixelscale)), int(round((self.origin[1] + self.shape[1] - obj.window.origin[1])/obj.pixelscale))))
        )

    def get_coordinate_meshgrid(self, pixelscale, x = 0., y = 0.):
        return np.meshgrid(
            np.linspace(self.origin[1] + pixelscale/2 - x, self.origin[1] + self.shape[1] - pixelscale/2 - x, int(round(self.shape[1]/pixelscale)), dtype=float),
            np.linspace(self.origin[0] + pixelscale/2 - y, self.origin[0] + self.shape[0] - pixelscale/2 - y, int(round(self.shape[0]/pixelscale)), dtype=float),
        )
        
    def get_data(self, image):
        return image.data[self.get_indices(image)]
        
    def scaled_window(self, scale, limit_window = None):
        # Determine the new shape by scaling the old one
        new_window_shape = self.shape * scale

        # New window origin from forcing the expected shape and center
        new_window_origin = self.center - new_window_shape / 2

        if limit_window is not None:# fixme origin and shape interaction
            new_window_origin = np.clip(new_window_origin, a_min = limit_window.origin, a_max = None)
            new_window_shape = np.clip(new_window_origin + new_window_shape, a_min = None, a_max = limit_window.origin + limit_window.shape) - new_window_origin
        
        return AP_Window(new_window_origin, new_window_shape)

    def buffer_window(self, buffer_size, limit_window = None):
        # Determine the new shape by adding to the old one
        new_window_shape = self.shape + 2 * buffer_size
        
        # New window origin from forcing the expected shape and center
        new_window_origin = self.center - new_window_shape / 2

        if limit_window is not None:
            new_window_origin = np.clip(new_window_origin, a_min = limit_window.origin, a_max = None)
            new_window_shape = np.clip(new_window_origin + new_window_shape, a_min = None, a_max = limit_window.origin + limit_window.shape) - new_window_origin
        
        return AP_Window(new_window_origin, new_window_shape)

    def __add__(self, other):
        new_origin = np.minimum(self.origin, other.origin)
        new_end = np.maximum(self.origin + self.shape, other.origin + other.shape)
        
        return AP_Window(new_origin, new_end - new_origin)
    
    def __iadd__(self, other):

        new_origin = np.minimum(self.origin, other.origin)
        new_end = np.maximum(self.origin + self.shape, other.origin + other.shape)

        self.origin = new_origin
        self.shape = new_end - new_origin
        self.center = self.origin + self.shape/2
        return self

    def __mul__(self, other):

        new_origin = np.maximum(self.origin, other.origin)
        new_end = np.minimum(self.origin + self.shape, other.origin + other.shape)

        return AP_Window(new_origin, new_end - new_origin)

    def __imul__(self, other):

        new_origin = np.maximum(self.origin, other.origin)
        new_end = np.minimum(self.origin + self.shape, other.origin + other.shape)

        self.origin = new_origin
        self.shape = new_end - new_origin
        self.center = self.origin + self.shape/2
        return self
        
    def __eq__(self, other):

        return all((np.all(self.origin == other.origin), np.all(self.shape == other.shape)))

    def __or__(self, other):

        overlap = self * other
        overlap_area = np.prod(overlap.shape)
        
        full_area = np.prod(self.shape) + np.prod(other.shape) - overlap_area
        
        return overlap_area / full_area

    def __str__(self):
        return f"window origin: {list(self.origin)}, shape: {list(self.shape)}, center: {list(self.center)}"
