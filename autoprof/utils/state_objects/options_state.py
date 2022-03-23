from .substate_object import SubState

class Options(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs

    def __get__(self, key):
        return self.options[key]

    def __contains__(self, key):
        return key in self.options
