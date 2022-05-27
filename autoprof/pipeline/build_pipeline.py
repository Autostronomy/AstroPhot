from functools import partial
from .default_pipelines import default_fitting_pipeline
from flow import Node, Chart
from autoprof import nodes

def build_pipeline(**options):
    ap_pipeline = default_fitting_pipeline
    if 'ap_pipeline_main' in options:
        ap_pipeline['structure'] = options['ap_pipeline_main']
    if 'ap_pipeline_fitloop' in options:
        ap_pipeline['node_kwargs']["fit_loop"]["structure"] = options['ap_pipeline_fitloop']
    if 'ap_pipeline' in options:
        ap_pipeline = options['ap_pipeline']

    if "ap_pipeline_insert_steps" in options:
        for step in options["ap_pipeline_insert_steps"]:
            pipe_index = ap_pipeline['structure'].index(step[0])
            ap_pipeline['structure'].insert(pipe_index, step[1])
    if "ap_pipeline_fitloop_insert_steps" in options:
        for step in options["ap_pipeline_fitloop_insert_steps"]:
            pipe_index = ap_pipeline['node_kwargs']['fit_loop']["structure"].index(step[0])
            ap_pipeline['node_kwargs']['fit_loop']["structure"].insert(pipe_index, step[1])
        
    # Loop through the keys of the pipeline dictionary, these represent flow.Charts and subcharts
    return Chart(**ap_pipeline)
