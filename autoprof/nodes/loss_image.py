from flow import Process

class Loss_Image(Process):
    """
    Compute a loss image by comparing the data image and model image
    """

    def action(self, state):

        # fixme image type
        state.loss_image = (state.image - state.model_image)**2

        return state
