from flow import Process

class Create_Models_Spec(Process):
    """
    Create models and add them to the state based on options specified by the user.
    """

    def action(self, state):
        for m in state.options['ap_models']:
            state.models.add_model(name = m, **state.options['ap_models'][m])

        return state


class Create_Models_SegMap(Process):
    """
    Create models and add them to the state based on options specified by the user and a provided segmentation map.
    """
    
