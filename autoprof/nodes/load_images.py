from .autoprof_node import make_AP_Process
from autoprof.image import AP_Image

@make_AP_Process("load images")
def load_images(state):
    """
    Load the image, sigma map, mask, and psf from memory which is to be fit by the model(s)
    """
    
    for input_image in ['image', 'psf', 'mask', 'sigma']:
        if f'ap_{input_image}_image' in state.options:
            img_kwargs = {
                'filename': state.options[f'ap_{input_image}_file'],
            }
            if f'ap_{input_image}_pixelscale' in state.options:
                img_kwargs['pixelscale'] = state.options[f'ap_{input_image}_pixelscale']
            else:
                img_kwargs['pixelscale'] = state.data.image.pixelscale
            if f'ap_{input_image}_index' in state.options:
                img_kwargs['index'] = state.options[f'ap_{input_image}_index']
            if f'ap_{input_image}_zeropoint' in state.options:
                img_kwargs['zeropoint'] = state.options[f'ap_{input_image}_zeropoint']
            if input_image == 'image':
                state.data.update_image(**img_kwargs)
            if input_image == 'psf':
                state.data.update_psf(**img_kwargs)
            if input_image == 'sigma':
                state.data.update_sigma(**img_kwargs)
            if input_image == 'mask':
                state.data.update_mask(**img_kwargs)
        
    
    return state
