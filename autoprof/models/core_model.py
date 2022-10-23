import torch
import numpy as np
import matplotlib.pyplot as plt

class AutoProf_Model(object):
    """Prototype class for all AutoProf models. Every class should
    define/overload the methods included here. The fit function is
    however, sufficient for most cases and likely shouldn't be
    overloaded.

    """
    
    learning_rate = 0.2
    max_iterations = 256
    stop_rtol = 1e-5

    def __init__(self, name, *args, **kwargs):
        self.name = name
    
    def initialize(self):
        """When this function finishes, all parameters should have numerical
        values (non None) that are reasonable estimates of the final
        values.

        """
        pass

    def finalize(self):
        """This is run after fitting and can be used to undo any fitting
        specific settings or aspects of a model.

        """
        pass

    def sample(self, sample_image = None):
        """Calling this function should fill the given image with values
        sampled from the given model.

        """
        pass

    def compute_loss(self):
        """Determine a realnumbered value to be minimized while optimizing
        model parameters.

        """
        pass

    def get_parameters_representation(self):
        """Get the optimizer friently representations of all the non-locked
        parameter values for this model.

        """
        pass
    
    def get_parameters_value(self):
        """Get the non-locked parameter values for this model.

        """
        pass
    
    def fit(self):
        """fit the model to data.

        """
        optimizer = torch.optim.Adam(self.get_parameters_representation(), lr = self.learning_rate)
        for epoch in range(self.max_iterations):
            self.epoch = epoch
            if (epoch % int(self.max_iterations/10)) == 0:
                print(f"{epoch}/{self.max_iterations}")
            optimizer.zero_grad()
            self.sample()
            
            plt.imshow(np.log10(self.model_image.data.detach().numpy()), vmax = 1.1, vmin = -5.9, origin = "lower")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames/sample_frame_{epoch:04d}.jpg", dpi = 400)
            plt.close()
            plt.imshow(self.target[self.fit_window].data.detach().numpy() - self.model_image.data.detach().numpy(), cmap = "seismic", vmax = 2., vmin = -2., origin = "lower")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames/sample_residual_frame_{epoch:04d}.jpg", dpi = 400)
            plt.close()
            self.compute_loss()
            self.loss.backward()
            
            start_params = []
            for p in self.get_parameters_representation():
                pv = p.detach().numpy()
                try:
                    float(pv)
                    start_params.append(pv)
                except:
                    start_params += list(pv)
            optimizer.step()
            step_params = []
            for p in self.get_parameters_representation():
                pv = p.detach().numpy()
                try:
                    float(pv)
                    step_params.append(pv)
                except:
                    step_params += list(pv)
            optimizer.zero_grad()
            if np.all(np.abs((np.array(start_params) / np.array(step_params)) - 1) < self.stop_rtol):
                print(epoch)
                break
        self.finalize()
        self.epoch = None
        
    def __str__(self):
        """String representation for the model.

        """
        return "Core Model Instance"
