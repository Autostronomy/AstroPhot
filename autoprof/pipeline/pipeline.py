from autoprof.nodes import *
from autoprof.pipeline.pipeline_construction.class_discovery import all_subclasses
from autoprof.utils.state_objects.state_object import State
from autoprof.models.model_object import Model
from functools import partial

class Pipeline(object):

    default_fitting_pipeline = {'main': ['load image', 'gaussian psf', "create models", 'initialize models', 'fit loop', 'quality checks', 'write models'],
                                'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'update parameters', 'stop iteration']}
    default_forced_pipeline = {'main': ['load image', 'gaussian psf', 'load_models', 'fit loop', 'quality checks', 'write models'],
                               'fit loop': ['sample models', 'project to image', 'select models', 'compute loss', 'update parameters', 'stop iteration']}
    def __init__(self):
        self.start_name = 'main'
    
    def build_pipeline(self, state):
        charts = {}
        if 'ap_pipeline' in state.options:
            ap_pipeline = state.options['ap_pipeline']
        elif Model.mode == 'fitting':
            ap_pipeline = self.default_fitting_pipeline
        elif Model.mode == 'forced':
            ap_pipeline = self.default_forced_pipeline
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

        return charts[self.start_name]

    def _process_state(self, state, chart = None):

        if chart:
            return chart(state)
        else:
            chart = self.build_pipeline(state)

        return chart(state)
        
    def __call__(self, chart = None, ap_mode = 'fitting', **options):
        state = State(**options)
        Model.mode = ap_mode
        
        if isinstance(state, State):
            return self._process_state(state, chart)
        else:
            return map(partial(self._process_state, chart = chart), state)
