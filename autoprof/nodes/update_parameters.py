from flow import Process
from autoprof.utils.optimization import k_delta_step
import numpy as np

class Update_Parameters_Random_Grad(Process):
    """
    Request loss history from each model and use this information to propose updated values for the parameters.
    Update the model parameters with a step that minimizes the loss using information from previous parameter-loss pairs.
    Stocastic kdelta operates by computing the gradient on the hyperplane defined by "k" samples of the loss function. A
    stocastic update is included to ensure the full parameter space can be explored even when k is less than dimensions
    plus one.
    """

    def action(self, state):

        state.models.step_iteration()
        # Loop through each model
        for model in state.models:
            # Skip locked models
            if model.locked:
                continue
            # Loop through each loss/parameters pairing
            for loss, params in model.get_loss_history(limit = 5):
                # Determine the perturbation scale
                param_scale = np.array(list(par.uncertainty for par in params[0]))

                # sample the random step
                update = np.random.normal(scale = param_scale)
                if len(loss) >= 5:
                    best = np.argmin(loss)
                else:
                    best = max(0, len(loss)-1)
                # Compute the gradient step
                if len(loss) >= 4:
                    grad_step = np.require(k_delta_step(loss, params, k = 3, reference = best),dtype=float)
                    update -= grad_step * model.learning_rate * np.linalg.norm(param_scale) / np.linalg.norm(grad_step)

                # Apply the update to each parameter
                for i, P in enumerate(params[0]):
                    P.set_representation(params[best][i].representation + update[i] * (1 + (model.learning_rate - 1) * np.sqrt(model.iteration/state.options.max_iterations)))
        
        return state

class Update_Parameters_Random(Process):
    """
    Request loss history from each model and use this information to propose updated values for the parameters.
    Update the model parameters with a random step starting from the parameter set out of the last 5 itterations
    which has the lowest loss.
    """

    def action(self, state):

        state.models.step_iteration()
        # Loop through each model
        for model in state.models:
            # Skip locked models
            if model.locked:
                continue
            # Loop through each loss/parameters pairing
            for loss, params in model.get_loss_history(limit = 5):
                # Determine the perturbation scale
                param_scale = np.array(list(par.uncertainty for par in params[0]))
                # sample the random step
                update = np.random.normal(scale = param_scale)
                # Identify the best sample of the last 5
                if len(loss) >= 5:
                    best = np.argmin(loss)
                else:
                    best = max(0, len(loss)-1)
                # Apply the update to each parameter
                for i, P in enumerate(params[0]):
                    P.set_representation(params[best][i].representation + update[i] * (1 + model.iteration * (model.learning_rate - 1)/1000))
                
        
        return state
