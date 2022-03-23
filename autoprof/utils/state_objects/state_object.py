from .data_state import Data
from .options_state import Options
from .results_state import Results
from .models_state import Models

class State(object):

    def __new__(cls, *args, **kwargs):

        if any(isinstance(v, list) for v in kwargs.values()):
            return (State(**sub_kwargs) for sub_kwargs in self._kwargs_iterator(**kwargs))
        else:
            return super().__new__(cls)
        
    def __init__(self, **kwargs):

        self.data = Data()
        self.results = Results()
        self.models = Models()
        self.options = Options(
            state = self,
            **kwargs
        )

    def _kwargs_iterator(self, **kwargs):

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
                    
