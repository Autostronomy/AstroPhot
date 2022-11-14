import torch
import numpy as np
from autoprof import plots
import matplotlib.pyplot as plt
from autoprof.utils.conversions.optimization import cyclic_difference_np
from autoprof.utils.conversions.dict_to_hdf5 import dict_to_hdf5
from copy import copy
from time import time

__all__ = ["AutoProf_Model"]

class AutoProf_Model(object):
    """Prototype class for all AutoProf models. Every class should
    define/overload the methods included here. The fit function is
    however, sufficient for most cases and likely shouldn't be
    overloaded.

    """

    model_type = ""
    learning_rate = 0.05
    max_iterations = 256
    stop_rtol = 1e-5
    constraint_delay = 10
    constraint_strength = 1e-2

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.epoch = None
        self.constraints = kwargs.get("constraints", None)
        self.is_sampled = False

    def initialize(self):
        """When this function finishes, all parameters should have numerical
        values (non None) that are reasonable estimates of the final
        values.

        """
        pass

    def startup(self):
        """Run immediately before fitting begins. Typically this is not
        needed, though it is available for models which require
        specific configuration for fitting, see also "finalize"

        """
        pass

    def finalize(self):
        """This is run after fitting and can be used to undo any fitting
        specific settings or aspects of a model.

        """
        pass

    def sample(self, sample_image=None):
        """Calling this function should fill the given image with values
        sampled from the given model.

        """
        pass

    def compute_loss(self):
        """Compute a standard Chi^2 loss given the target image, model, and
        variance image. Typically if overloaded this will also be
        called with super() and higher methods will multiply or add to
        the loss.

        """
        pixels = np.prod(self.target[self.fit_window].data.shape)
        if self.target.masked:
            mask = np.logical_not(self.target[self.fit_window].mask)
            self.loss = torch.sum(
                torch.pow(
                    (
                        self.target[self.fit_window].data[mask]
                        - self.model_image.data[mask]
                    ),
                    2,
                )
                / self.target[self.fit_window].variance[mask]
            )
            pixels -= np.sum(mask)
        else:
            self.loss = torch.sum(
                torch.pow((self.target[self.fit_window] - self.model_image).data, 2)
                / self.target[self.fit_window].variance
            )
        self.loss /= pixels - len(self.get_parameters_representation()[0])
        if (
            self.constraints is not None
            and self.epoch is not None
            and self.epoch > self.constraint_delay
        ):
            for constraint in self.constraints:
                self.loss *= 1 + self.constraint_strength * (
                    self.epoch - self.constraint_delay
                ) * constraint(self)
        print("loss: ", self.loss)
        return self.loss

    def step(self):
        """Call after updating any of the model parameters.

        """
        self.is_sampled = False
        
    def get_parameters_representation(self):
        """Get the optimizer friently representations of all the non-locked
        parameter values for this model.

        """
        pass

    def get_parameters_value(self):
        """Get the non-locked parameter values for this model."""
        pass

    def fit(self):
        """Iteratively fit the model to data."""
        self.startup()
        self.step()
        keys, reps = self.get_parameters_representation()
        optimizer = torch.optim.Adam(
            reps, lr=self.learning_rate, betas = (0.9,0.9),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.2, patience=20, min_lr = self.learning_rate*1e-2, cooldown = 10)
        loss_history = []
        start = time()
        for epoch in range(self.max_iterations):
            self.epoch = epoch
            if (epoch % int(self.max_iterations / 10)) == 0:
                print(f"Epoch: {epoch}/{self.max_iterations}.")
                if epoch > 0:
                    print(f"Est time to completion: {(self.max_iterations - epoch)*(time() - start)/(epoch*60)} min")

            optimizer.zero_grad()
            self.sample()
            self.compute_loss()
            loss_history.append(copy(self.loss.detach().item()))
            # if (len(loss_history) - np.argmin(loss_history)) > 20:
            #     print(epoch)
            #     break
            
            self.loss.backward()
            
            skeys, sreps = self.get_parameters_representation()
            for i in range(len(sreps)):
                if not np.all(np.isfinite(sreps[i].grad.detach().numpy())):
                    print("WARNING: nan grad being fixed")
                    sreps[i].grad *= 0
            optimizer.step()
            scheduler.step(self.loss)
            self.step()
        self.finalize()
        print("runtime: ", time() - start)
        self.epoch = None
        plt.plot(range(len(loss_history)), np.log10(loss_history))
        plt.show()

    def __str__(self):
        """String representation for the model."""
        return "AutoProf Model Instance"

    def get_state(self):
        state = {
            "name": self.name,
            "model type": self.model_type,
        }
        return state

    def save(self, filename = "AutoProf.yaml"):
        if filename.endswith(".yaml"):
            import yaml
            state = self.get_state()
            with open(filename, "w") as f:
                yaml.dump(state, f)            
        elif filename.endswith(".json"):
            import json
            state = self.get_state()
            with open(filename, "w") as f:
                json.dump(state, f, indent = 2)
        elif filename.endswith(".hdf5"):
            import h5py
            state = self.get_state()
            with h5py.File(filename, "w") as F:
                dict_to_hdf5(F, state)
        else:
            if isinstance(filename, str) and '.' in filename:
                raise ValueError(f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5")
            else:
                raise ValueError(f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5")
            
    def load(self, filename = "AutoProf.yaml"):
        if filename.endswith(".yaml"):
            import yaml
            with open(filename, "r") as f:
                state = yaml.load(f)            
        elif filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                state = json.load(f)
        elif filename.endswith(".hdf5"):
            import h5py
            with h5py.File(filename, "r") as F:
                state = hdf5_to_dict(F)
        elif isinstance(filename, dict):
            state = filename
        else:
            if isinstance(filename, str) and '.' in filename:
                raise ValueError(f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5")
            else:
                raise ValueError(f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5 or python dictionary.")
        return state
