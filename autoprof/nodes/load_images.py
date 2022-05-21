from flow import Process

class Load_Images(Process):
    """
    Load the image, sigma map, mask, and psf from memory which is to be fit by the model(s)
    """

    def action(self, state):
        for input_image in ['target', 'psf', 'mask', 'sigma']:
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
                if input_image == 'target':
                    state.data.update_image(**img_kwargs)
                elif input_image == 'psf':
                    state.data.update_psf(**img_kwargs)
                elif input_image == 'sigma':
                    state.data.update_sigma(**img_kwargs)
                elif input_image == 'mask':
                    state.data.update_mask(**img_kwargs)
        
        return state
