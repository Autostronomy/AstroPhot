from .. import AP_config
from .base import Node

__all__ = ["Group_Node"]
    
class Group_Node(Node):
    
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs) 

    @property
    def value(self):
        return self.flat_value(include_locked = False)

    @value.setter
    def value(self, val):
        if self.locked and not Node.global_unlock:
            return
        self._set_val_subnodes(val)
        self.shape = None
        
    def _set_val_subnodes(self, val):
        flat = self.flat(include_locked = False)
        loc = 0
        for node in flat.keys():
            node.value = val[loc:loc + node.size]
            loc += node.size

