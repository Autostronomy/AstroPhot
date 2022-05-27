from flow import Process

class Sample_Models(Process):
    """
    Create a model image based on the current set of model parameters.
    """

    def action(self, state):
        state.data.initialize_model_image()
        state.models.sample_models()
        state.models.convolve_psf()

        for model in state.models:
            state.data.model_image += model.model_image
            
        return state

