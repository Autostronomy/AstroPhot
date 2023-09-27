from collections import OrderedDict
from abc import ABC, abstractmethod

__all__ = ["Node"]

class Node(ABC):
    global_unlock = False
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.nodes = OrderedDict()
        self.locked = kwargs.get("locked", False)

    def link(self, *nodes):
        for node in nodes:
            self.nodes[node.name] = node

    def unlink(self, *nodes):
        for node in nodes:
            del self.nodes[node.name]

    def dump(self):
        self.unlink(*self.nodes.values())

    def __getitem__(self, key):
        if key in self.nodes:
            return self.nodes[key]
        base, stem = key.split(":", 1)
        return self.nodes[base][stem]

    def __contains__(self, key):
        return key in self.nodes

    def get_state(self):
        state = {
            "name": self.name,
            "identity": id(self),
        }
        if len(self.nodes) > 0:
            state["nodes"] = tuple(node.get_state() for node in self.nodes.values())
        return state

    def set_state(self, state):
        self.name = state["name"]

        if "nodes" in state:
            for node in state["nodes"]:
                self.link(self.__class__(**node))

    @property
    @abstractmethod
    def value(self):
        ...

    def flat(self, include_locked = True):
        flat = OrderedDict()
        for node in self.nodes.values():
            if len(node.nodes) == 0 and not node.value is None:
                if node.locked and not (include_locked or Node.global_unlock):
                    continue
                flat[node] = None
            else:
                flat.update(node.flat(include_locked))
        return flat

    def flat_value(self, include_locked = False):
        flat = self.flat(include_locked)
        size = 0
        for node in flat.keys():
            size += node.size

        val = torch.zeros(size, dtype=AP_config.ap_dtype, device=AP_config.ap_device)
        loc = 0
        for node in flat.keys():
            val[loc:loc + node.size] = node.value.flatten()
            loc += node.size
        return val

    def __str__(self):
        return f"Node: {self.name}"
    def __repr__(self):
        return f"Node: {self.name} " + ("locked" if self.locked else "unlocked")
