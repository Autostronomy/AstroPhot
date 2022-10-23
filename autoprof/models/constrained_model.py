from .super_model_object import Super_Model

class Constrained_Model(Super_Model):
    """A version of Super_Model which can apply arbitrary parameter
    constraints on the model parameters.  Takes an argument
    "constraints" which is a list of functions. Each function takes as
    argument the model object and returns a scalar which is zero when
    the constraint is satisfied and continuously increases if the
    constraint is violated. For example, to add the constraint that
    "x" is equal between two models, the function would be:
    
    def constraint_equal(model):
        return torch.abs(model["model1|x"].value - model["model2|x"].value)
    
    which is zero only when the two parameters are equal.

    """
    constraint_delay = 10
    constraint_strength = 1e-1
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.constraints = kwargs.get("constraints", None)
        
    def compute_loss(self):
        super().compute_loss()
        if self.constraints is not None and self.epoch > self.constraint_delay:
            for constraint in self.constraints:
                self.loss += self.constraint_strength * (self.epoch - self.constraint_delay) * constraint(self)
        return self.loss
