import numpy as np
from autoprof.utils.conversions.optimization import boundaries, inv_boundaries, cyclic_boundaries, cyclic_difference
from copy import deepcopy

class Parameter(object):

    def __init__(self, name, **kwargs):

        self.name = name
        
        self.limits = kwargs.get("limits", None)
        self.cyclic = kwargs.get("cyclic", False)
        self.user_fixed = kwargs.get("fixed", None)
        self.update_fixed(False)
        self.value = None
        self.representation = None
        if "value" in kwargs:
            self.set_value(kwargs["value"], override_fixed = True)
        self.units = kwargs.get("units", "none")
        self.uncertainty = kwargs.get("uncertainty", None)

    def update_fixed(self, fixed):
        self.fixed = fixed or bool(self.user_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if np.any(uncertainty < 0):
            raise ValueError(f"{name} Uncertainty should be a positive real value, not {uncertainty}")
        self.uncertainty = uncertainty
        
    def set_value(self, value, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.cyclic:
            self.value = cyclic_boundaries(value, self.limits)
            self.representation = self.value
            return
        self.value = value
        if self.limits is None:
            self.representation = self.value
        else:
            assert self.limits[0] is None or value > self.limits[0]
            assert self.limits[1] is None or value < self.limits[1]
            self.representation = boundaries(self.value, self.limits)
        
    def set_representation(self, representation, override_fixed = False):
        if self.fixed and not override_fixed:
            return
        if self.limits is None or self.cyclic:
            self.set_value(representation, override_fixed)
        else:
            self.set_value(inv_boundaries(representation, self.limits), override_fixed)

    def __str__(self):
        return f"{self.name}: {self.value} +- {self.uncertainty} [{self.units}{'' if self.fixed is False else ', fixed'}{'' if self.limits is None else (', ' + str(self.limits))}{'' if self.cyclic is False else ', cyclic'}]"

    def __sub__(self, other):

        if self.cyclic:
            return cyclic_difference(self.representation, other.representation, self.limits[1] - self.limits[0])

        return self.representation - other.representation

class Parameter_Array(Parameter):
    
    def set_value(self, value, override_fixed = False, index = None):
        if self.value is None:
            self.value = []
            for i, val in enumerate(value):
                self.value.append(Parameter(
                    name = f"{self.name}:{i}",
                    limits = self.limits,
                    cyclic = self.cyclic,
                    fixed = self.user_fixed,
                    value = val,
                    units = self.units,
                    uncertainty = self.uncertainty
                ))
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_value(value[i], override_fixed)
        else:
            self.value[index].set_value(value, override_fixed)

    def get_values(self):
        return np.array(list(V.value for V in self.value))
        
    def set_representation(self, representation, override_fixed = False, index = None):
        
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_representation(representation[i], override_fixed)
        else:
            self.value[index].set_representation(representation, override_fixed)

    def set_uncertainty(self, uncertainty, override_fixed = False, index = None):
        if index is None:
            for i in range(len(self.value)):
                self.value[i].set_uncertainty(uncertainty[i], override_fixed)
        else:
            self.value[index].set_uncertainty(uncertainty, override_fixed)
        
        
    def __sub__(self, other):
        res = np.zeros(len(self.value))
        for i in range(len(self.value)):
            if isinstance(other, Parameter_Array):
                res[i] = self.value[i] - other.value[i]
            elif isinstance(other, Parameter):
                res[i] = self.value[i] - other
            else:
                raise ValueError(f"unrecognized parameter type: {type(other)}")
            
        return res

    def __iter__(self):
        self.i = -1
        return self
    def __next__(self):
        self.i += 1
        if self.i < len(self.value):
            return self[self.i]
        else:
            raise StopIteration
    
    def __getitem__(self, S):
        try:
            return self.value[S]
        except KeyError:
            for v in self.value:
                if S == v.name:
                    return v
            raise KeyError(f"{S} not in {self.name}. {str(self)}")

    def __str__(self):
        return "\n".join([f"{self.name}:"] + list(str(val) for val in self.value))
        
    def __len__(self):
        return len(self.value)


class Optimize_History(object):

    def __init__(self, model):
        self.name = model.name
        self.parameter_history = []
        self.loss_history = []
        self.map_loss_quality = {"global": set()}
        for param in model.parameters:
            if "loss" in model.parameter_qualities[param]:
                if model.parameter_qualities[param]["loss"] in self.map_loss_quality:
                    self.map_loss_quality[model.parameter_qualities[param]["loss"]].add(param)
                else:
                    self.map_loss_quality[model.parameter_qualities[param]["loss"]] = set([param])
            else:
                self.map_loss_quality["global"].add(param)

    def add_step(self, params, loss):
        self.parameter_history.insert(0, deepcopy(params))
        self.loss_history.insert(0, deepcopy(loss))

    def get_parameters(self, index=0, exclude_fixed = False, quality = None):
        """
        index: index of the parameter history where 0 is most recent and -1 is first
        exclude_fixed: ignore parameters currently set to fixed
        quality: select for a parameter quality, should be a tuple of length 2. first element is the quality name, second is the desired value.
        """
        use_params = self.parameter_history[index]
        
        # Return all parameters for a given iteration
        if not exclude_fixed and quality is None:
            return self.parameter_history[index]
        return_parameters = {}
        for p in self.parameter_history[index]:
            # Skip currently fixed parameters since they cannot be updated anyway
            if (exclude_fixed and self.parameter_history[index][p].fixed) or (quality is not None and p not in self.map_loss_quality[quality]):
                continue
            # Return representation which is valid in [-inf, inf] range
            return_parameters[p] = self.parameter_history[index][p]
        return return_parameters

    def get_loss(self, index=0, parameter = None, loss_quality = "global"):
        """
        Return a loss value for this model.
        index: index of the loss history where 0 is most recent and -1 is first
        parameter: return the loss associated with this parameter, defaults to "global"
        loss_quality: directly request a specific loss calculation, defaults to "global"
        """
        loss_dict = self.loss_history[index]
        
        if parameter is not None:
            for mlq in self.map_loss_quality:
                if parameter in self.map_loss_quality[mlq]:
                    return loss_dict[mlq]
            else:
                return loss_dict["global"]
        return loss_dict[loss_quality]
    
    def get_loss_history(self, limit = np.inf):
        loss_history = {}
        for loss_quality in self.map_loss_quality.keys():
            # All global parameters
            param_order = self.get_parameters(exclude_fixed = True, quality = ["loss", loss_quality]).keys()
            
            # handle loss vector instances
            if not isinstance(self.get_loss(loss_quality = loss_quality), float):
                for il in range(len(self.get_loss(loss_quality = loss_quality))):
                    params = []
                    loss = []
                    for i in range(min(limit, len(self.loss_history))):
                        params_i = self.get_parameters(index = i, exclude_fixed = True, quality = ["loss", loss_quality])
                        sub_params = []
                        for P in param_order:
                            if isinstance(params_i[P], Parameter_Array):
                                if self.parameters[P][il].fixed:
                                    continue
                                sub_params.append(params_i[P][il])
                            elif isinstance(params_i[P], Parameter):
                                sub_params.append(params_i[P])
                        params.append(np.array(sub_params))
                        loss.append(self.get_loss(index = i, loss_quality = loss_quality)[il])
                    loss_history[f"{loss_quality}:{il}"] = (loss, params)
                continue

            # handle regular loss values
            params = []
            loss = []
            for i in range(min(limit, len(self.loss_history))):
                params_i = self.get_parameters(index = i, exclude_fixed = True, quality = ["loss", loss_quality])
                sub_params = []
                for P in param_order:
                    if isinstance(params_i[P], Parameter_Array):
                        for ip in range(len(params_i[P])):
                            if self.parameters[P][ip].fixed:
                                continue
                            sub_params.append(params_i[P][ip])
                    elif isinstance(params_i[P], Parameter):
                        sub_params.append(params_i[P])
                params.append(np.array(sub_params))
                loss.append(self.get_loss(index = i, loss_quality = loss_quality))
            loss_history[loss_quality] = (loss, params)
        return loss_history
        
        
