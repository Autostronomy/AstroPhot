import torch
import numpy as np
from autoprof import plots
import matplotlib.pyplot as plt
from autoprof.utils.conversions.optimization import cyclic_difference_np
from autoprof.utils.conversions.dict_to_hdf5 import dict_to_hdf5
from copy import copy
from time import time
from torch.autograd.functional import jacobian
__all__ = ["AutoProf_Model"]

def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])

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

    dtype = torch.float32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __new__(cls, *args, **kwargs):
        if "filename" in kwargs:
            state = AutoProf_Model.load(kwargs["filename"])
            MODELS = all_subclasses(AutoProf_Model)
            for M in MODELS:
                if M.model_type == state["model_type"]:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(f"Unknown AutoProf model type: {state['model_type']}")
        elif "model_type" in kwargs:
            MODELS = all_subclasses(AutoProf_Model)
            for M in MODELS:
                if M.model_type == kwargs["model_type"]:
                    return super(AutoProf_Model, cls).__new__(M)
            else:
                raise ModuleNotFoundError(f"Unknown AutoProf model type: {kwargs['model_type']}")
            
        return super().__new__(cls)

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
        self.parameter_vector_len = list(int(np.prod(self[P].value.shape)) for P in self.parameter_order)
        
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
        pixels = torch.prod(self.model_image.window.shape/self.target.pixelscale)
        if self.target.masked:
            mask = torch.logical_not(self.target[self.model_image.window].mask)
            pixels -= torch.sum(mask)
            self.loss = torch.sum(
                torch.pow(
                    (
                        self.target[self.model_image.window].data[mask]
                        - self.model_image.data[mask]
                    ),
                    2,
                )
                / self.target[self.model_image.window].variance[mask]
            )
        else:
            self.loss = torch.sum(
                torch.pow((self.target[self.model_image.window] - self.model_image).data, 2)
                / self.target[self.model_image.window].variance
            )
        self.loss /= pixels - len(self.get_parameters_representation()[0])
        if self.constraints is not None:
            for constraint in self.constraints:
                self.loss *= 1 + self.constraint_strength * constraint(self)
        return self.loss

    def step(self, parameters = None, parameters_as_representation = True):
        """Call after updating any of the model parameters.

        """
        self.is_sampled = False
        if parameters is None:
            return
        if isinstance(parameters, dict):
            for P in parameters:
                isinstance(parameters[P], torch.Tensor)
                if parameters_as_representation:
                    self[P].representation = parameters[P]
                else:
                    self[P].value = parameters[P]
            return
        assert isinstance(parameters, torch.Tensor)
        start = 0
        for P, V in zip(self.parameter_order, self.parameter_vector_len):
            if parameters_as_representation:
                self[P].representation = parameters[start:start + V].reshape(self[P].representation.shape)
            else:
                self[P].value = parameters[start:start + V].reshape(self[P].value.shape)
            start += V
        
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
        optimizer = torch.optim.NAdam(
            reps, lr=0.1, betas = (0.9,0.999),
        )
        # optimizer = torch.optim.LBFGS(
        #     reps, lr=0.05,
        # )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        loss_history = []
        start = time()
        for epoch in range(self.max_iterations):
            self.epoch = epoch
            if (epoch % int(self.max_iterations / 10)) == 0:
                print(f"Epoch: {epoch}/{self.max_iterations}.")
                if epoch > 0:
                    print(f"Est time to completion: {(self.max_iterations - epoch)*(time() - start)/(epoch*60):.2f} min")
                    print(f"X^2/ndf: {self.loss.detach().cpu().item():.3e}")
                    
            optimizer.zero_grad()
                
            self.sample()
            self.compute_loss()            
            self.loss.backward()
            loss_history.append(copy(self.loss.detach().cpu().item()))
            
            skeys, sreps = self.get_parameters_representation()
            optimizer.step()
            if epoch % 10 == 0 and epoch > 0:
                scheduler.step()
            self.step()
        self.finalize()
        print(f"Runtime: {(time() - start)/60:.2f} min")
        print(f"X^2/ndf: {self.loss.detach().cpu().item():.3e}")
        self.epoch = None
        plt.plot(range(len(loss_history))[10:], np.log10(loss_history)[10:])
        plt.savefig("loss_history.jpg")
        plt.close()

    def full_sample(self, parameters = None):
        self.step(parameters)
        self.sample()
        self.compute_loss()
        return self.model_image.data

    def forward(self, parameters):
        return self.full_sample(parameters)
    
    def full_loss(self, parameters = None):
        self.step(parameters)
        self.sample()
        self.compute_loss()
        return self.loss

    def jacobian(self, parameters):
        return jacobian(
            self.full_sample,
            parameters,
            strategy = "forward-mode",
            vectorize = True,
            create_graph = False,
        )

    def __str__(self):
        """String representation for the model."""
        return "AutoProf Model Instance"

    def get_state(self):
        state = {
            "name": self.name,
            "model_type": self.model_type,
        }
        return state

    def save(self, filename = "AutoProf.yaml"):
        if filename.endswith(".yaml"):
            import yaml
            state = self.get_state()
            with open(filename, "w") as f:
                yaml.dump(state, f, indent = 2) 
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

    @classmethod
    def load(cls, filename = "AutoProf.yaml"):
        if isinstance(filename, dict):
            state = filename
        elif filename.endswith(".yaml"):
            import yaml
            with open(filename, "r") as f:
                state = yaml.load(f, Loader = yaml.FullLoader)            
        elif filename.endswith(".json"):
            import json
            with open(filename, 'r') as f:
                state = json.load(f)
        elif filename.endswith(".hdf5"):
            import h5py
            with h5py.File(filename, "r") as F:
                state = hdf5_to_dict(F)
        else:
            if isinstance(filename, str) and '.' in filename:
                raise ValueError(f"Unrecognized filename format: {filename[filename.find('.'):]}, must be one of: .json, .yaml, .hdf5")
            else:
                raise ValueError(f"Unrecognized filename format: {str(filename)}, must be one of: .json, .yaml, .hdf5 or python dictionary.")
        return state
        
