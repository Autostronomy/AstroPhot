from .substate_object import SubState
from autoprof.models import Model
from autoprof.pipeline.class_discovery import all_subclasses
from autoprof.image import Model_Image
import numpy as np
import matplotlib.pyplot as plt
from autoprof.diagnostic_plots.shared_elements import LSBImage


class Models_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.models = {}
        self.model_list = []
        self.model_image = None
        
    def add_model(self, name, model, **kwargs):
        MODELS = all_subclasses(Model)
        if isinstance(model, str):
            for m in MODELS:
                if m.model_type == model:
                    self.models[name] = m(name, self.state, self.state.data.image, **kwargs)
                    break
        elif isinstance(model, Model):
            self.models[name] = model(name, self.state, self.state.data.image, **kwargs)
        else:
            raise ValueError('model should be a string or AutoProf Model object, not: {type(model)}')

        self.model_list.append(name)
        self.organize_model_list()
        
    def organize_model_list(self):
        model_sizes = list(-(self.models[m].model_image.shape[0] * self.models[m].model_image.shape[1]) for m in self.model_list)
        N = np.argsort(model_sizes)
        new_list = []
        for n in N:
            new_list.append(self.model_list[n])
        self.model_list = new_list
            
    def initialize(self):
        
        for m in self.model_list:
            self.models[m].initialize()

    def compute_loss(self, loss_image):
        
        for m in self.model_list:
            self.models[m].compute_loss(loss_image)

    def sample_models(self):
        self.model_image = Model_Image(
            np.zeros(self.state.data.image.shape),
            pixelscale=self.state.data.image.pixelscale,
            origin=self.state.data.image.origin,
        )
        
        for m in self.model_list:
            self.models[m].sample_model()
            self.model_image.add_image(self.models[m].model_image)

    def step_iteration(self):
        for m in self.model_list:
            self.models[m].step_iteration()

    def save_models(self, filename):
        lims = (np.min(self.model_image), np.max(self.model_image))
        plt.imshow(np.log10(self.model_image - lims[0] + 0.01*(lims[1]-lims[0])), origin = 'lower')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename[:filename.find('.')+1] + 'jpg', dpi = 400)
        plt.close()
        with open(filename, "w") as f:
            for m in self.model_list:
                self.models[m].save_model(f)
            
    def __iter__(self):
        self._iter_model = -1
        return self

    def __next__(self):
        self._iter_model += 1
        if self._iter_model < len(self.model_list):
            return self.models[self.model_list[self._iter_model]]
        else:
            raise StopIteration
