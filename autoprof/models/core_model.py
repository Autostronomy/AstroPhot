

class AutoProf_Model(object):

    def __init__(self, name, *args, **kwargs):
        self.name = name
    
    def initialize(self):
        pass

    def finalize(self):
        pass

    def sample(self):
        pass

    def compute_loss(self):
        pass

    def __str__(self):
        return "Core Model Instance"
