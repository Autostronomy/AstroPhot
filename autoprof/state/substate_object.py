    
class SubState(object):

    def __init__(self, **kwargs):
        self.state = kwargs['state']
        del kwargs['state']
    
