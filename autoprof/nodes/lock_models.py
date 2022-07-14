from flow import Process
import numpy as np

class Lock_Models(Process):
    """
    Choose models that seem to have converged and lock them.
    """

    def action(self, state):

        N_check = 128
        for model in state.models:
            if model.locked or model.iteration == 0:
                continue
            
            if model.iteration >= N_check and model.iteration % (2*N_check) == 0:
                loss_history = model.history.get_loss_history(2*N_check, quality = "global")
                best = np.argmin(loss_history)
                if best > N_check:
                    model.update_locked(2*N_check+1)
            # if not model.locked and model.iteration == state.models.iteration and np.std(model.loss_history[:100]) < (np.min(model.loss_history[:100])/1e3):
            #     print("locking: ", model.name)
            #     model.update_locked(200)
            #     continue
            
            # If a new best loss hasnt been found for 50 iterations then assume it has converged
            # min_so_far = np.min(model.loss_history[100:200])
            # min_close = np.min(model.loss_history[10:20])
            # if not model.locked and np.all((min_so_far - np.array(model.loss_history[:100])) < (min_so_far / 100)) and np.all((min_close - np.array(model.loss_history[:10])) < (min_close / 100)):
            #     print("locking: ", model.name)
            #     model.update_locked(True)
            #     continue
            # If the model has been locked for a long time, unlock it to check
            # if model.locked and (state.models.iteration - model.iteration) > 200:
            #     model.update_locked(False)
                
        return state
