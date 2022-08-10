from flow import Process

class Sample_Models(Process):
    """
    Create a model image based on the current set of model parameters.
    """

    def action(self, state):
        state.data.initialize_model_image()
        state.models.sample_models()

        for model in state.models:
            state.data.model_image += model.model_image
            
        return state

class Sample_Expanded_Models(Process):
    """
    Expand the model windows and sample the model parameters to get more complete result.
    """

    def action(self, state):

        # Scale all the model windows to the new size
        scale_factor = state.options["ap_sample_expanded_models_scale", 3]
        for model in state.models:
            model.scale_window(scale_factor)
        state.models.sample_models()

        full_target = state.options["ap_sample_expanded_models_fulltarget", False]
        include_locked = state.options["ap_sample_expanded_models_includelocked", False]
        state.data.initialize_model_image(full_target = full_target, include_locked = include_locked)
        # Build the model image
        for model in state.models:
            state.data.model_image += model.model_image
        
        return state
