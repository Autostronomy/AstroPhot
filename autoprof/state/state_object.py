from .data_state import Data_State
from .options_state import Options_State
from .results_state import Results_State
from .models_state import Models_State

class State(object):

    def __new__(cls, *args, **kwargs):

        if any(isinstance(v, list) for v in kwargs.values()):
            return (State(**sub_kwargs) for sub_kwargs in cls._kwargs_iterator(**kwargs))
        else:
            return super().__new__(cls)
        
    def __init__(self, **kwargs):

        self.data = Data_State(state = self)
        self.results = Results_State(state = self)
        self.models = Models_State(state = self)
        self.options = Options_State(
            state = self,
            **kwargs
        )

    def _kwargs_iterator(**kwargs):

        for v in kwargs.values():
            if isinstance(v,list):
                num_states = len(v)
                break
        for i in range(num_states):
            sub_kwargs = {}
            for k in kwargs:
                if isinstance(kwargs[k],list):
                    sub_kwargs[k] = kwargs[k][i]
                else:
                    sub_kwargs[k] = kwargs[k]
            yield sub_kwargs
                    
