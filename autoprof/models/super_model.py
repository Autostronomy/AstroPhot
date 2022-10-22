from .core_model import AutoProf_Model
from autoprof.image import Target_Image, Model_Image
from copy import deepcopy
import torch
import numpy as np
import matplotlib.pyplot as plt

class Super_Model(AutoProf_Model):
    learning_rate = 0.2
    max_iterations = 256
    stop_rtol = 1e-5
    
    def __init__(self, name, model_list, target = None, locked = False, **kwargs):
        super().__init__(name, model_list, target, **kwargs)
        self.model_list = model_list
        self.target = self.model_list[0].target if target is None else target
        self._user_locked = locked
        self._locked = self._user_locked
        self.loss = None
        self.update_fit_window()

    def add_model(self, model):
        self.model_list.append(model)
        self.update_fit_window()
        
    def update_fit_window(self):
        self.fit_window = None
        for model in self.model_list:
            if model.locked:
                continue
            if self.fit_window is None:
                self.fit_window = deepcopy(model.fit_window)
            else:
                self.fit_window |= model.fit_window
        self.model_image = Model_Image(
            pixelscale = self.target.pixelscale,
            window = self.fit_window,
        )
        
    def initialize(self):
        for model in self.model_list:
            model.initialize()
    def initialize(self):
        for model in self.model_list:
            model.locked = True
        self.sample()
        plt.imshow(np.log10(self.model_image.data.detach().numpy()), vmax = 1.1, vmin = -5.9, origin = "lower")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"frames/init_frame_{0:04d}.jpg", dpi = 400)
        plt.close()
        plt.imshow(self.target[self.fit_window].data.detach().numpy() - self.model_image.data.detach().numpy(), cmap = "seismic", vmax = 2., vmin = -2., origin = "lower")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"frames/init_residual_frame_{0:04d}.jpg", dpi = 400)
        plt.close()
        for mi, model in enumerate(self.model_list):
            model.locked = False
            model.initialize()
            self.sample()
            plt.imshow(np.log10(self.model_image.data.detach().numpy()), vmax = 1.1, vmin = -5.9, origin = "lower")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames/init_frame_{mi+1:04d}.jpg", dpi = 400)
            plt.close()
            plt.imshow(self.target[self.fit_window].data.detach().numpy() - self.model_image.data.detach().numpy(), cmap = "seismic", vmax = 2., vmin = -2., origin = "lower")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"frames/init_residual_frame_{mi+1:04d}.jpg", dpi = 400)
            plt.close()

    def finalize(self):
        for model in self.model_list:
            model.finalize()
        
    def sample(self, sample_image = None):
        if self.locked:
            return
        if sample_image is None or sample_image is self.model_image:
            self.model_image.clear_image()
        self.loss = None
        
        for model in self.model_list:
            if model.locked:
                continue
            model.sample(sample_image)
            if sample_image is None:
                self.model_image += model.model_image
        
    def compute_loss(self):
        self.loss = torch.sum(torch.pow((self.target[self.fit_window] - self.model_image).data, 2) / self.target[self.fit_window].variance)
        return self.loss

    def fit(self):
        optimizer = torch.optim.Adam(self.get_parameters_representation(), lr = self.learning_rate)
        for epoch in range(self.max_iterations):
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
                        
    def get_parameters_representation(self, exclude_locked = True):
        all_parameters = []
        for model in self.model_list:
            all_parameters += model.get_parameters_representation(exclude_locked)
        return all_parameters
    
    def get_parameters_value(self, exclude_locked = True):
        all_parameters = {}
        for model in self.model_list:
            values = model.get_parameters_value(exclude_locked)
            for p in values:
                all_parameters[f"{model.name}|{p}"] = values[p] 
        return all_parameters
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.model_list[key[0]][key[1]]

        if isinstance(key, str) and "|" in key:
            return self.model_list[int(key[:key.find("|")])][key[key.find("|")+1:]]
        
        raise KeyError(f"{key} not in {self.name}. {str(self)}")

    @property
    def target(self):
        return self._target
    @target.setter
    def target(self, tar):
        assert isinstance(tar, Target_Image)
        self._target = tar
        for model in self.model_list:
            model.target = tar
            
    from ._model_methods import locked
