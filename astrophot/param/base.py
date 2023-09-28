from collections import OrderedDict
from abc import ABC, abstractmethod

__all__ = ["Node"]

class Node(ABC):
    global_unlock = False
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.nodes = OrderedDict()
        if "state" in kwargs:
            self.set_state(kwargs["state"])
            return
        if "nodes" in kwargs:
            for node in kwargs["nodes"]:
                self.link(self.__class__(**node))
        self.locked = kwargs.get("locked", False)

    def link(self, *nodes):
        for node in nodes:
            for subnode_id in node.flat().keys():
                if self.identity == subnode_id:
                    raise RuntimeError("Parameter structure must be Directed Acyclic Graph! Adding this node would create a cycle")
            self.nodes[node.name] = node
            
    def unlink(self, *nodes):
        for node in nodes:
            del self.nodes[node.name]

    def dump(self):
        self.unlink(*self.nodes.values())

    def __getitem__(self, key):
        if key == self.name:
            return self
        if key in self.nodes:
            return self.nodes[key]
        if isinstance(key, str) and ":" in key:
            base, stem = key.split(":", 1)
            return self.nodes[base][stem]
        if isinstance(key, int):
            for node in self.nodes.values():
                if key == node.identity:
                    return node
        raise ValueError(f"Unrecognized key for '{self}': {key}")
                
    def __contains__(self, key):
        return key in self.nodes

    def __eq__(self, other):
        return self is other

    @property
    def identity(self):
        try:
            return self._identity
        except AttributeError:
            return id(self)
    
    def get_state(self):
        state = {
            "name": self.name,
            "identity": self.identity,
        }
        if self.locked:
            state["locked"] = self.locked
        if len(self.nodes) > 0:
            state["nodes"] = list(node.get_state() for node in self.nodes.values())
        return state

    def set_state(self, state):
        self.name = state["name"]
        self._identity = state["identity"]
        if "nodes" in state:
            for node in state["nodes"]:
                self.link(self.__class__(**node))
        self.locked = state.get("locked", False)
    def __iter__(self):
        return filter(lambda n: not n.locked, self.nodes.values())
    
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
                flat[node.identity] = node
            else:
                flat.update(node.flat(include_locked))
        return flat


    def __str__(self):
        return f"Node: {self.name}"
    def __repr__(self):
        return f"Node: {self.name} " + ("locked" if self.locked else "unlocked")
