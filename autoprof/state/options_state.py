from .substate_object import SubState

class Options_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs

    def __getitem__(self, key):
        try:
            return self.options[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return key in self.options
