from flow import Process
from autoprof.utils.optimization import k_delta_step, local_derivative
import numpy as np

class Update_Parameters_Random_Grad(Process):
    """
    Request loss history from each model and use this information to propose updated values for the parameters.
    Update the model parameters with a step that minimizes the loss using information from previous parameter-loss pairs.
    Stocastic kdelta operates by computing the gradient on the hyperplane defined by "k" samples of the loss function. A
    stocastic update is included to ensure the full parameter space can be explored even when k is less than dimensions
    plus one.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scheduler = {}
    
    def action(self, state):

        N_best = 32
        state.models.step_iteration()
        # Loop through each model
        for model in state.models:
            # Skip locked models
            if model.locked or model.iteration == 0:
                continue

            loss_history, param_history = model.get_history(limit = max(N_best, len(model.get_parameters(exclude_fixed = True))+1))

            # Pick which loss keys are going to be optimized
            if len(loss_history) == 0:
                print("skipping")
                continue
            # elif len(loss_history) == 1:
            #     run_loss = list(loss_history.keys())[0]
            # else:
            #     if model.name not in self.loss_scheduler:
            #         self.loss_scheduler[model.name] = []
            #         for loss_type in loss_history:
            #             loss_type_base = loss_type.split(" ")[0]
            #             if not loss_type_base in self.loss_scheduler[model.name]:
            #                 self.loss_scheduler[model.name].append(loss_type_base)
            #     run_loss = self.loss_scheduler[model.name][(max(0,model.iteration-1) // N_best) % len(self.loss_scheduler[model.name])]
            # run_losses = []
            # for key in loss_history.keys():
            #     if key.split(" ")[0] == run_loss:
            #         run_losses.append(key)
                    
            # for loss, params in list(loss_history[key] for key in run_losses):
            # If all params are fixed, skip this optimization step
            best = np.argmin(loss_history[:N_best])
            if len(loss_history) >= 2:
                for par in range(len(param_history[0])):
                    if best == 0:
                        model[param_history[0][par].name].uncertainty *= 1.11
                    else:
                        model[param_history[0][par].name].uncertainty *= 0.98
            # Determine the perturbation scale
            param_scale = np.array(list(model[par.name].uncertainty for par in param_history[0]))

            update = np.zeros(len(param_scale))
            
            # sample the random step
            update += np.random.normal(scale = param_scale)
                
            # Compute the gradient step
            if len(loss_history) >= (len(param_history[0])+1) and model.iteration > (state.options.max_iterations/2):
                grad_step = np.require(local_derivative(param_history[:(len(param_history[0])+1)], loss_history[:(len(param_history[0])+1)]),dtype=float)
                grad_norm = np.linalg.norm(grad_step)
                if grad_norm > 1e-5:
                    update -= 0.1 * grad_step * np.linalg.norm(param_scale) / grad_norm
                        
            # Apply the update to each parameter
            for i, P in enumerate(param_history[0]):
                model[P.name].set_representation(param_history[best][i].representation + update[i])
            
        return state

class Update_Parameters_Random(Process):
    """
    Request loss history from each model and use this information to propose updated values for the parameters.
    Update the model parameters with a random step starting from the parameter set out of the last 5 itterations
    which has the lowest loss.
    """

    def action(self, state):

        N_best = 32
        state.models.step_iteration()
        # Loop through each model
        for model in state.models:
            # Skip locked models
            if model.locked or model.iteration == 0:
                continue
            loss_history, param_history = model.get_history(limit = max(N_best, len(model.get_parameters(exclude_fixed = True))+1))
            best = np.argmin(loss_history[:N_best])
            # Determine the perturbation scale
            param_scale = np.array(list(model[par.name].uncertainty for par in param_history[0]))
            if len(loss_history) >= 2:
                for par in range(len(param_history[0])):
                    if best == 0:
                        model[param_history[0][par].name].uncertainty *= 1.11
                    else:
                        model[param_history[0][par].name].uncertainty *= 0.98

            update = np.zeros(len(param_scale))
            
            # sample the random step
            update += np.random.normal(scale = param_scale)
            # Apply the update to each parameter
            for i, P in enumerate(param_history[0]):
                model[P.name].set_representation(param_history[best][i].representation + update[i])
        
        return state
