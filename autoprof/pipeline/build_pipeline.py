from autoprof.nodes import *
from autoprof.models.model_object import Model
from .class_discovery import all_subclasses
from functools import partial

default_fitting_pipeline = {'main': ['load image', 'gaussian psf', "create models", 'initialize models', 'fit loop', 'quality checks', 'save models'],
                            'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'update parameters', 'stop iteration']}
default_forced_pipeline = {'main': ['load image', 'gaussian psf', 'load models', 'fit loop', 'quality checks', 'save models'],
                           'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'update parameters', 'stop iteration']}
    
def build_pipeline(**options):
    charts = {}
    if 'ap_mode' in options:
        Model.mode = options['ap_mode']
    if 'ap_pipeline' in options:
        ap_pipeline = options['ap_pipeline']
    elif Model.mode == 'fitting':
        ap_pipeline = default_fitting_pipeline
    elif Model.mode == 'forced':
        ap_pipeline = default_forced_pipeline
    else:
        raise ValueError(f'Unrecognized mode: {Model.mode}')
            
    # Discover all nodes that have been imported
    NODES = all_subclasses(AP_Node)
    # Loop through the keys of the pipeline dictionary, these represent flow.Charts and subcharts
    for chart_name in ap_pipeline:
        # Create the chart
        charts[chart_name] = AP_Chart(chart_name)
            
    # Loop through the keys of the pipeline dictionary, these represent flow.Charts and subcharts
    for chart_name in ap_pipeline:
        # Initiate linear mode, this simply means that each node will be added one at a time to the flow.Chart
        charts[chart_name].linear_mode(True)
        # Loop through all nodes associated with this chart (value of the dictionary is list of strings)
        for node in ap_pipeline[chart_name]:
            assert node != chart_name
            if node in charts:
                charts[chart_name].add_node(charts[node])
                continue
            # Check all nodes that exist for the one specified in the dictionary
            for n in NODES:
                # Find the right node
                if n.name == node:
                    # Create a new node and add it to the chart
                    charts[chart_name].add_node(n(name = f"{chart_name}:{node}"))
                    break
            else:
                raise ValueError(f'Unrecognized node: {node}')
                    
        charts[chart_name].linear_mode(False)

    return charts['main']
