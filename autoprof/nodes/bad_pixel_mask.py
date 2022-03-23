from ..autoprof_node import make_AP_Process

@make_AP_Process("bad pixel mask")
def bad_pixel_mask(state):
    """
    Mask the image based on specified bad pixels. Pixels above an overflow value or below a floor value, or equal (plus/minus some epsilon) to a key value.
    """
    return state
