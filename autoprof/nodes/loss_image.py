from flow import Process

class Loss_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):

        # fixme image type

        state.data.loss_image = state.data.target - state.data.model_image
        state.data.loss_image.data = state.data.loss_image.data**2

        return state
