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
        
    def overlap_frac(self, other):
        overlap = self & other
        overlap_area = np.prod(overlap.shape)
        full_area = np.prod(self.shape) + np.prod(other.shape) - overlap_area
        return overlap_area / full_area

    # Window adjustment operators
    def __add__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_origin = self.origin - other
            new_shape = self.shape + 2*other
            return AP_Window(new_origin, new_shape)
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            new_origin = self.origin - np.array(other)
            new_shape = self.shape + 2*np.array(other)
            return AP_Window(new_origin, new_shape)
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __iadd__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.origin -= other
            self.shape += 2*other
            return self
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            self.origin -= np.array(other)
            self.shape += 2*np.array(other)
            return self
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __sub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_origin = self.origin - other
            new_shape = self.shape + 2*other
            return AP_Window(new_origin, new_shape)
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            new_origin = self.origin - np.array(other)
            new_shape = self.shape + 2*np.array(other)
            return AP_Window(new_origin, new_shape)
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __isub__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.origin += other
            self.shape -= 2*other
            return self
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            self.origin += np.array(other)
            self.shape -= 2*np.array(other)
            return self
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_shape = self.shape * other
            new_origin = self.center - new_shape / 2
            return AP_Window(new_origin, new_shape)
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            new_shape = self.shape * np.array(other)
            new_origin = self.center - new_shape / 2
            return AP_Window(new_origin, new_shape)
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __imul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.shape *= other
            self.origin = self.center - new_window_shape / 2
            return self
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            self.shape *= np.array(other)
            self.origin = self.center - new_window_shape / 2
            return self
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __div__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            new_shape = self.shape / other
            new_origin = self.center - new_shape / 2
            return AP_Window(new_origin, new_shape)
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            new_shape = self.shape / np.array(other)
            new_origin = self.center - new_shape / 2
            return AP_Window(new_origin, new_shape)
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")
    def __idiv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            self.shape /= other
            self.origin = self.center - new_window_shape / 2
            return self
        elif (isinstance(other, tuple) or isinstance(other, np.ndarray)) and len(other) == len(self.origin):
            self.shape /= np.array(other)
            self.origin = self.center - new_window_shape / 2
            return self
        raise ValueError(f"AP_Window object cannot be added with {type(other)}")

    # Window Comparison operators
    def __eq__(self, other):
        return all((np.all(self.origin == other.origin), np.all(self.shape == other.shape)))
    def __neq__(self, other):
        return not self == other
    def __gt__(self, other):
        return np.all(self.origin < other.origin) and np.all((self.origin + self.shape) > (other.origin + other.shape))
    def __gtq__(self, other):
        return np.all(self.origin <= other.origin) and np.all((self.origin + self.shape) >= (other.origin + other.shape))
    def __lt__(self, other):
        return np.all(self.origin > other.origin) and np.all((self.origin + self.shape) < (other.origin + other.shape))
    def __ltq__(self, other):
        return np.all(self.origin >= other.origin) and np.all((self.origin + self.shape) <= (other.origin + other.shape))

    # Window interaction operators
    def __or__(self, other):
        new_origin = np.minimum(self.origin, other.origin)
        new_end = np.maximum(self.origin + self.shape, other.origin + other.shape)
        return AP_Window(new_origin, new_end - new_origin)
    def __ior__(self, other):
        new_origin = np.minimum(self.origin, other.origin)
        new_end = np.maximum(self.origin + self.shape, other.origin + other.shape)
        self.origin = new_origin
        self.shape = new_end - new_origin
        self.center = self.origin + self.shape/2
        return self
    def __and__(self, other):
        new_origin = np.maximum(self.origin, other.origin)
        new_end = np.minimum(self.origin + self.shape, other.origin + other.shape)
        return AP_Window(new_origin, new_end - new_origin)
    def __iand__(self, other):
        new_origin = np.maximum(self.origin, other.origin)
        new_end = np.minimum(self.origin + self.shape, other.origin + other.shape)
        self.origin = new_origin
        self.shape = new_end - new_origin
        self.center = self.origin + self.shape/2
        return self
        
    def __str__(self):
        return f"window origin: {list(self.origin)}, shape: {list(self.shape)}, center: {list(self.center)}"
