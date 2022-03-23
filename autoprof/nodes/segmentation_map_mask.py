from .autoprof_node import make_AP_Process

@make_AP_Process("segmentation map mask")
def segmentation_map_mask(state):
    """
    Mask pixels based on a segmentation map, the map can identify all pixels that should be avoided while fitting. Any segments which identify models that are being fit will be ignored.
    """
    return state
