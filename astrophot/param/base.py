from collections import OrderedDict

__all__ = ["Node"]

class Node(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self.nodes = OrderedDict()

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
        return self.nodes[key[:base]][stem]

    def get_state(self):
        state = {
            "name": self.name,
            "identity": id(self),
        }
        if len(self.nodes) > 0:
            state["nodes"] = tuple(node.get_state() for node in self.nodes.values())
        return state
