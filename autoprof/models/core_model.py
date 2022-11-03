import torch
import numpy as np
import matplotlib.pyplot as plt
from autoprof.utils.conversions.optimization import cyclic_difference_np
from copy import copy
from time import time

__all__ = ["AutoProf_Model"]

class AutoProf_Model(object):
    """Prototype class for all AutoProf models. Every class should
    define/overload the methods included here. The fit function is
    however, sufficient for most cases and likely shouldn't be
    overloaded.

    """

    learning_rate = 0.2
    max_iterations = 256
    stop_rtol = 1e-5
    constraint_delay = 10
    constraint_strength = 1e-1

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
        print("learning rate: ", self.learning_rate)
        keys, reps = self.get_parameters_representation()
        optimizer = torch.optim.Adam(
            reps, lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience=5, min_lr = self.learning_rate*1e-2, cooldown = 10)
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
            if (len(loss_history) - np.argmin(loss_history)) > 20:
                print(epoch)
                break
            
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
        self.epoch = None
        plt.plot(range(len(loss_history)), np.log10(loss_history))
        plt.show()

    def __str__(self):
        """String representation for the model."""
        return "AutoProf Model Instance"
