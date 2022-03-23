from flow import Node, Process, Decision, Chart
from weakref import WeakSet
import functools

class AP_Node(Node):

    instances = WeakSet()
    name = 'AP_Node'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

        AP_Node.instances.add(self)        

class AP_Process(Process, AP_Node):
    name = 'AP_Process'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

class AP_Decision(Decision, AP_Node):
    name = 'AP_Decision'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

class AP_Chart(Chart, AP_Node):
    name = 'AP_Chart'
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)


def make_AP_Process(new_name):
    def inner(func):
        class New_Process(AP_Process):
            name = new_name
            def __init__(self, *args, **kwargs):
                if 'name' in kwargs:
                    node_name = kwargs['name']
                    del kwargs['name']
                else:
                    node_name = new_name
                super().__init__(*args, name = node_name, **kwargs)
                self.update_run(func)
            #new_node = AP_Process(name = name, func = func)
        return New_Process
    return inner

def make_AP_Decision(new_name):
    def inner(func):
        class New_Decision(AP_Decision):
            name = new_name
            def __init__(self, *args, **kwargs):
                if 'name' in kwargs:
                    node_name = kwargs['name']
                    del kwargs['name']
                else:
                    node_name = new_name
                super().__init__(*args, name = node_name, **kwargs)
                self.update_run(func)
            #new_node = AP_Decision(name = name, func = func)
        return New_Decision
    return inner
