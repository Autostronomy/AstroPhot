from .substate_object import SubState
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

    def __getitem__(self, key):
        try:
            return self.options[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return key in self.options
