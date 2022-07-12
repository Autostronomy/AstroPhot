from .substate_object import SubState
from autoprof.models import BaseModel
from autoprof.pipeline.class_discovery import all_subclasses
import os

class Options_State(SubState):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options = kwargs

        if 'ap_saveto' in self.options:
            self.save_path = self.options['ap_saveto']
        else:
            self.save_path = ""
        
        if "ap_plot_path" in self.options:
            self.plot_path = self.options['ap_plot_path']
        elif 'ap_saveto' in self.options:
            self.plot_path = self.options['ap_saveto']
        else:
            self.plot_path = ""

        if "ap_name" in self.options:
            self.name = self.options['ap_name']
        elif "ap_target_file" in self.options:
            self.name = os.path.splitext(os.path.basename(self.options['ap_target_file']))[0]
        else:
            self.name = "AutoProfModel"

        self.max_iterations = kwargs.get("ap_max_iterations", 1000)

        if "ap_model_atributes" in self.options:
            print("setting attributes")
            # Get all model subclasses
            MODELS = all_subclasses(BaseModel)
            # Loop through models specified by users
            for identifier in self.options["ap_model_atributes"]:
                # Match with model subclasses
                for model in MODELS:
                    if model.model_type == identifier:
                        # Apply updates to parameters as specified by user
                        for attr in self.options["ap_model_atributes"][identifier]:
                            # Skip attributes that cannot be updated
                            if attr in ['model_type']:
                                continue
                            setattr(model, attr, self.options["ap_model_atributes"][identifier][attr])

    def __getitem__(self, key):
        try:
            return self.options[key]
        except KeyError:
            try:
                return self.options[key[0]]
            except KeyError:
                return key[1]

    def __contains__(self, key):
        return key in self.options
