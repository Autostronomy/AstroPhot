from collections import OrderedDict
from abc import ABC, abstractmethod

__all__ = ["Node"]

class Node(ABC):
    """Base node object in the Directed Acyclic Graph (DAG).

    The base Node object handles storing the DAG nodes and links
    between them. An important part of the DAG system is to be able to
    find all the leaf nodes, which is done using the `flat` function.

    """
    global_unlock = False
    
    def __init__(self, name, **kwargs):
        if ":" in name:
            raise ValueError(f"Node names must not have ':' character. Cannot use name: {name}")
        self.name = name
        self.nodes = OrderedDict()
        if "state" in kwargs:
            self.set_state(kwargs["state"])
            return
        if "link" in kwargs:
            self.link(*kwargs["link"])
        self.locked = kwargs.get("locked", False)

    def link(self, *nodes):
        """Creates a directed link from the current node to the provided
        node(s) in the input. This function will also check that the
        linked node does not exist higher up in the DAG to the current
        node, if that is the case then a cycle has formed which breaks
        the DAG structure and could cause problems. An error will be
        thrown in this case.

        The linked node is added to a ``nodes`` dictionary that each
        node stores. This makes it easy to check which nodes are
        linked to each other.

        """
        for node in nodes:
            for subnode_id in node.flat(include_locked=True, include_links=True).keys():
                if self.identity == subnode_id:
                    raise RuntimeError("Parameter structure must be Directed Acyclic Graph! Adding this node would create a cycle")
            self.nodes[node.name] = node
            
    def unlink(self, *nodes):
        """Undoes the linking of two nodes. Note that this could sever the
        connection of many nodes to each other if the current node was
        the only link between two branches.

        """
        for node in nodes:
            del self.nodes[node.name]

    def dump(self):
        """Simply unlinks all nodes that the current node is linked with.

        """
        self.unlink(*self.nodes.values())

    @property
    def leaf(self):
        """Returns True when the current node is a leaf node.

        """
        return len(self.nodes) == 0

    def __getitem__(self, key):
        """Used to get a node from the DAG relative to the current node. It
        is possible to collect nodes from deeper in the DAG by
        separating the names of the nodes along the path with a colon
        (:). For example::

          first_node["second_node:third_node"]

        returns a node that is actually linked to ``second_node``
        without needing to first get ``second_node`` then call
        ``second_node['third_node']``.

        """
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
                self.link(self.__class__(name = node["name"], state = node))
        self.locked = state.get("locked", False)
    def __iter__(self):
        return filter(lambda n: not n.locked, self.nodes.values())
    
    @property
    @abstractmethod
    def value(self):
        ...

    def flat(self, include_locked = True, include_links = False):
        flat = OrderedDict()
        if self.leaf and self.value is not None:
            if not self.locked or include_locked or Node.global_unlock:
                flat[self.identity] = self
        for node in self.nodes.values():
            if node.leaf and node.value is not None:
                if node.locked and not (include_locked or Node.global_unlock):
                    continue
                flat[node.identity] = node
            else:
                if include_links and (not node.locked or include_locked or Node.global_unlock):
                    flat[node.identity] = node
                flat.update(node.flat(include_locked))
        return flat


    def __str__(self):
        return f"Node: {self.name}"
    def __repr__(self):
        return f"Node: {self.name} " + ("locked" if self.locked else "unlocked") + ("" if self.leaf else " {" + ";".join(repr(node) for node in self.nodes) + "}")
