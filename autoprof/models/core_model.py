import torch
import numpy as np
import matplotlib.pyplot as plt

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
        else:
            self.loss = torch.sum(
                torch.pow((self.target[self.fit_window] - self.model_image).data, 2)
                / self.target[self.fit_window].variance
            )

        if (
            self.constraints is not None
            and self.epoch is not None
            and self.epoch > self.constraint_delay
        ):
            for constraint in self.constraints:
                self.loss *= 1 + self.constraint_strength * (
                    self.epoch - self.constraint_delay
                ) * constraint(self)

        return self.loss

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
        optimizer = torch.optim.Adam(
            self.get_parameters_representation(), lr=self.learning_rate
        )
        for epoch in range(self.max_iterations):
            self.epoch = epoch
            if (epoch % int(self.max_iterations / 10)) == 0:
                print(f"{epoch}/{self.max_iterations}")
            optimizer.zero_grad()
            self.sample()

            plt.imshow(
                np.log10(self.model_image.data.detach().numpy()),
                vmax=1.1,
                vmin=-5.9,
                origin="lower",
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames_late/sample_frame_{epoch:04d}.jpg", dpi=400)
            plt.close()
            plt.imshow(
                self.target[self.fit_window].data.detach().numpy()
                - self.model_image.data.detach().numpy(),
                cmap="seismic",
                vmax=2.0,
                vmin=-2.0,
                origin="lower",
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames_late/sample_residual_frame_{epoch:04d}.jpg", dpi=400)
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
            print(
                "rtol: ",
                np.max(
                    np.abs(np.abs(np.array(start_params) / np.array(step_params)) - 1)
                ),
                (np.array(start_params) / np.array(step_params)),
            )
            if np.allclose(start_params, step_params, rtol=self.stop_rtol, atol=0.0):
                print(epoch)
                break
        self.finalize()
        self.epoch = None

    def __str__(self):
        """String representation for the model."""
        return "AutoProf Model Instance"
