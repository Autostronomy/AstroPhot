from flow import Process
from autoprof.utils.calculations.optimization import stocastic_k_delta_step
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
            for loss, params in model.get_loss_history(limit = 4):
                # Determine the perturbation scale
                param_scale = np.array(list(par.uncertainty for par in params[0]))

                # Compute the gradient step
                if len(loss) < 4:
                    update = np.random.normal(scale = param_scale)
                else:
                    update = model.learning_rate * stocastic_k_delta_step(loss, params, param_scale)

                # Apply the update to each parameter
                for i, P in enumerate(params[0]):
                    P.set_representation(P.representation - update[i])
                
        
        return state
