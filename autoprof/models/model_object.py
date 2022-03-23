try:
    import cPickle as pickle
except:
    import pickle

class Model(object):

    PSF_mode = 'none'
    mode = 'fitting'
    
    def __init__(self, **kwargs):

        if 'load_file' in kwargs:
            self.load(kwargs['load_file'])
            return
        self.image_window = None
        self.image = None
        self.model_image = None
        self.parameters = [{}]
        self.loss = []
        self.fixed = set()
        self.limits = {}
        self.iteration = -1
        self.stage = 'created'
        
        if 'image_window' in kwargs:
            self.image_window = kwargs['image_window']
        if 'image' in kwargs:
            self.image = kwargs['image']
        if 'parameters' in kwargs:
            self.parameters[0].update(kwargs['parameters'])
        if 'fixed' in kwargs:
            self.fixed.update(kwargs['fixed'])
        if 'limits' in kwargs:
            self.limits.update(kwargs['limits'])
            
    def initialize(self):
        self.stage = 'initialized'

    def step_iteration(self):
        self.parameters.insert(0, self.parameters[0])
        self.iteration += 1
        self.stage = 'fitting'
        
    def update_loss(self, loss_image):
        return
    
    def sample_model(self):
        return

    def write(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            self.__dict__ = pickle.load(f)

    def get_iteration(self):
        return self.iteration
    
    def get_loss(self, index = 0):
        return self.loss[index]
    
    def get_parameters_representation(self, index = 0):
        return_parameters = {}
        for p in self.parameters[index]:
            if p in self.fixed:
                continue
            return_parameters[p] = self.get_representation(p, index)
            
    def set_parameters_representation(self, parameters):
        for p in parameters:
            self.set_representation(p, parameters[p])
            
    def update_parameters_representation(self, parameters):
        for p in parameters:
            self.add_representation(p, parameters[p])
            
    def get_value(self, key, index = 0):
        return self.parameters[index][key]
    
    def set_value(self, key, value, override_fixed = False):
        if key in self.fixed and not override_fixed:
            return
        self.parameters[0][key] = value

    def get_representation(self, key, index = 0):
        if key in self.limits:
            return boundaries(self.get_value(key, index), self.limits[key])
        else:
            return self.get_value(key, index)
        
    def set_representation(self, key, value, override_fixed = False):
        if key in self.limits:
            self.set_value(key, inv_boundaries(value, self.limits[key]))
        else:
            self.set_value(key, value)
            
    def add_representation(self, key, value):
        self.set_representation(key, self.get_representation(key) + value)
    
    def __get__(self, key, index = 0):
        return self.get_value(key, index)
