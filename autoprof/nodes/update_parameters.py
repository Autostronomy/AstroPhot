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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_scheduler = {}
    
    def action(self, state):

        N_uncertainty = 32
        N_lim = 8
        N_check = 128
        state.models.step_iteration()
        # Loop through each model
        for model in state.models:
            # Skip locked models
            if model.locked or model.iteration == 0:
                continue
            print(model.name)

            if model.iteration >= N_check and model.iteration % N_check == 0:
                loss_history = model.history.get_loss_history(N_check, quality = "global")
                best = np.argmin(loss_history)
                if best > N_check/2:
                    print("!!!!older is better!!!!")
                    old_params = model.history.get_parameters(best)
                    for P in old_params:
                        model[P].set_representation(old_params[P].get_representation())
                    continue
            
            # Loop through each loss/parameters pairing
            if model.iteration >= N_uncertainty and model.iteration % N_uncertainty == 0:
                uncertainty_update = True
            else:
                uncertainty_update = False

            loss_history = model.history.get_history(limit = N_uncertainty if uncertainty_update else min(N_lim, 1 + model.iteration % N_uncertainty))

            # Pick which loss keys are going to be optimized
            if len(loss_history) == 0:
                continue
            elif len(loss_history) == 1:
                run_loss = list(loss_history.keys())[0]
            else:
                if model.name not in self.loss_scheduler:
                    self.loss_scheduler[model.name] = []
                    for loss_type in loss_history:
                        loss_type_base = loss_type.split(" ")[0]
                        if not loss_type_base in self.loss_scheduler[model.name]:
                            self.loss_scheduler[model.name].append(loss_type_base)
                run_loss = self.loss_scheduler[model.name][(max(0,model.iteration-1) // N_uncertainty) % len(self.loss_scheduler[model.name])]
            run_losses = []
            for key in loss_history.keys():
                if key.split(" ")[0] == run_loss:
                    run_losses.append(key)
                    
            for loss, params in list(loss_history[key] for key in run_losses):
                # If all params are fixed, skip this optimization step
                if len(params) == 0 or loss is None:
                    continue

                if uncertainty_update:
                    for par in range(len(params[0])):
                        reps = list(P[par].representation for P in params)
                        if np.abs(np.mean(reps[:N_uncertainty//2]) - np.mean(reps[N_uncertainty//2:])) > (np.std(reps[:N_uncertainty//2])/np.sqrt(N_uncertainty//2)):
                            model[params[0][par].name].uncertainty *= 1.1
                        else:
                            model[params[0][par].name].uncertainty *= 0.7
                        
                # Determine the perturbation scale
                param_scale = np.array(list(model[par.name].uncertainty for par in params[0]))

                # sample the random step
                update = np.random.normal(scale = param_scale)
                best = np.argmin(loss[:N_lim])
                
                # Compute the gradient step
                if len(loss) >= N_lim:
                    grad_step = np.require(k_delta_step(loss[:N_lim], params[:N_lim], k = N_lim - 1, reference = best),dtype=float)
                    grad_norm = np.linalg.norm(grad_step)
                    if grad_norm > 1e-5:
                        update -= grad_step * model.learning_rate * np.linalg.norm(param_scale) / grad_norm

                # Apply the update to each parameter
                for i, P in enumerate(params[0]):
                    model[P.name].set_representation(params[best][i].representation + update[i])
        
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
            for loss, params in model.get_history(limit = 5):
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
