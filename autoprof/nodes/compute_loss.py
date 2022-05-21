from flow import Process

class Compute_Loss(Process):
    """
    Call each model to compute it's loss based on the loss image
    """

    def action(self, state):
        state.models.compute_loss(state.data.loss_image)
        return state
