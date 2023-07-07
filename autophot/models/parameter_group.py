from typing import Optional
from collections import OrderedDict

import torch
import numpy as np

from ..utils.conversions.optimization import (
    boundaries,
    inv_boundaries,
    d_boundaries_dval,
    d_inv_boundaries_dval,
    cyclic_boundaries,
)
from .parameter_object import Parameter
from .. import AP_config

__all__ = ["Parameter_Group"]


class Parameter_Group(object):
    def __init__(self, name, groups=None, parameters=None, state=None):
        self.name = name
        self.groups = OrderedDict()
        self.top_level_parameters = set()
        self.parameters = OrderedDict()
        if state is not None:
            self.set_state(state)
            return
        if groups is not None:
            for G in groups:
                self.add_group(G)
        if parameters is not None:
            for P in parameters:
                self.add_parameter(P)

    def copy(self):
        return Parameter_Group(
            name=self.name,
            groups=list(group.copy() for group in self.groups.values()),
            parameters=list(parameter.copy() for parameter in self.parameters.values()),
        )

    def add_group(self, group):
        self.groups[group.name] = group
        for P in group.parameters.values():
            self.add_parameter(P, top_level=False)

    def add_parameter(self, parameter, top_level=True):
        if top_level:
            self.top_level_parameters.update([parameter.identity])

        # Add the parameter identity to the list of parameters in this group
        self.parameters[parameter.identity] = parameter
        # Add a link for the parameter back to this group
        parameter.groups.add(self)

        # Redundant parameter setting, ensures that parameters with the same identity do not coexist (generally only applicable when loading a model)
        for group in self.groups.values():
            if parameter.identity in group.parameters:
                group.add_parameter(parameter, top_level=False)

    def get_identities(self):
        return sum((list(param.identities) for param in self), [])

    def order(self, parameters_identity=None):
        if parameters_identity is None:
            return list(param.identity for param in self)
        else:
            ret_list = []
            IDs = self.get_identities()
            for param in self:
                if any(pid in parameters_identity for pid in param.identities):
                    ret_list.append(param.identity)
            return ret_list

    def vector_len(self, parameters_identity=None):
        param_vec_len = []
        for P in self.order(parameters_identity=parameters_identity):
            if parameters_identity is None:
                param_vec_len.append(int(np.prod(self.get_id(P).value.shape)))
            else:
                param_vec_len.append(
                    sum(pid in parameters_identity for pid in self.get_id(P).identities)
                )
        return param_vec_len

    def get_vector(self, as_representation=False, parameters_identity=None):
        PVL = self.vector_len(parameters_identity=parameters_identity)
        parameters = torch.zeros(
            int(np.sum(PVL)),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        porder = self.order(parameters_identity=parameters_identity)

        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid in self.get_id(P).identities:
                    if pid in parameters_identity:
                        if as_representation:
                            parameters[pindex] = self.get_id(P).get_representation(
                                identity=pid
                            )
                        else:
                            parameters[pindex] = self.get_id(P).get_value(identity=pid)
                        pindex += 1
            return parameters

        # If the full vector is requested, they are added in bulk
        vstart = 0
        for P, V in zip(porder, PVL):
            if as_representation:
                parameters[vstart : vstart + V] = self.get_id(P).representation
            else:
                parameters[vstart : vstart + V] = self.get_id(P).value
            vstart += V
        return parameters

    def get_identity_vector(self, parameters_identity=None):
        parameters = []
        porder = self.order(parameters_identity=parameters_identity)

        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid in self.get_id(P).identities:
                    if pid in parameters_identity:
                        parameters.append(pid)
            return parameters

        # If the full vector is requested, they are added in bulk
        for P in porder:
            parameters += list(self.get_id(P).identities)
        return parameters

    def get_name_vector(self, parameters_identity=None):
        PVL = self.vector_len(parameters_identity=parameters_identity)
        parameters = list(None for i in range(int(sum(PVL))))
        porder = self.order(parameters_identity=parameters_identity)

        # If vector is requested by identity, they are individually updated
        pindex = 0
        if parameters_identity is not None:
            for P in porder:
                isarray = len(self.get_id(P).identities) > 1
                for ind, pid in enumerate(self.get_id(P).identities):
                    if pid in parameters_identity:
                        if isarray:
                            parameters[pindex] = f"{self.get_id(P).name}:{ind}"
                        else:
                            parameters[pindex] = self.get_id(P).name
                        pindex += 1
            return parameters

        # If the full vector is requested, they are added in bulk
        for P, V in zip(porder, PVL):
            isarray = len(self.get_id(P).identities) > 1
            if isarray:
                for ind, pid in enumerate(self.get_id(P).identities):
                    parameters[pindex] = f"{self.get_id(P).name}:{ind}"
                    pindex += 1
            else:
                parameters[pindex] = self.get_id(P).name
                pindex += 1
        return parameters
        
    
    def transform(
        self, in_parameters, to_representation=True, parameters_identity=None
    ):
        PVL = self.vector_len(parameters_identity=parameters_identity)
        out_parameters = torch.zeros(
            int(np.sum(PVL)),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        porder = self.order(parameters_identity=parameters_identity)

        # If vector is requested by identity, they are individually updated
        if parameters_identity is not None:
            pindex = 0
            for P in porder:
                for pid in self.get_id(P).identities:
                    if pid in parameters_identity:
                        if to_representation:
                            out_parameters[pindex] = self.get_id(P).val_to_rep(
                                in_parameters[pindex]
                            )
                        else:
                            out_parameters[pindex] = self.get_id(P).rep_to_val(
                                in_parameters[pindex]
                            )
                        pindex += 1
            return out_parameters

        # If the full vector is requested, they are added in bulk
        vstart = 0
        for P, V in zip(porder, PVL):
            if to_representation:
                out_parameters[vstart : vstart + V] = self.get_id(P).val_to_rep(
                    in_parameters[vstart : vstart + V]
                )
            else:
                out_parameters[vstart : vstart + V] = self.get_id(P).rep_to_val(
                    in_parameters[vstart : vstart + V]
                )
            vstart += V
        return out_parameters

    def get_uncertainty_vector(self, as_representation=False):
        PVL = self.vector_len()
        uncertanty = torch.zeros(
            int(np.sum(PVL)),
            dtype=AP_config.ap_dtype,
            device=AP_config.ap_device,
        )
        vstart = 0
        for P, V in zip(
            self.order(),
            PVL,
        ):
            if as_representation:
                uncertanty[vstart : vstart + V] = self.get_id(
                    P
                ).uncertainty_representation
            else:
                uncertanty[vstart : vstart + V] = self.get_id(P).uncertainty
            vstart += V
        return uncertanty

    def set_values(
        self,
        values,
        as_representation=True,
        parameters_identity=None,
    ):
        # ensure parameters are a tensor
        values = torch.as_tensor(
            values, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        # track the order of the parameters
        porder = self.order(parameters_identity=parameters_identity)

        # If parameters are provided by identity, they are individually updated
        if parameters_identity is not None:
            parameters_identity = list(parameters_identity)
            for P in porder:
                for pid in self.get_id(P).identities:
                    if pid in parameters_identity:
                        if as_representation:
                            self.get_id(P).set_representation(
                                values[parameters_identity.index(pid)], identity=pid
                            )
                        else:
                            self.get_id(P).set_value(
                                values[parameters_identity.index(pid)], identity=pid
                            )
            return

        # If parameters are provided as the full vector, they are added in bulk
        start = 0
        for P, V in zip(
            porder,
            self.vector_len(),
        ):
            if as_representation:
                self.get_id(P).representation = values[start : start + V].reshape(
                    self.get_id(P).representation.shape
                )
            else:
                self.get_id(P).value = values[start : start + V].reshape(
                    self.get_id(P).value.shape
                )
            start += V

    def get_values_as_tuple(self, as_representation=False, parameters_identity=None):
        if as_representation:
            return tuple(
                self.parameters[identity].representation
                for identity in self.order(parameters_identity=parameters_identity)
            )
        return tuple(
            self.parameters[identity].value
            for identity in self.order(parameters_identity=parameters_identity)
        )

    def set_values_from_tuple(
        self, values, as_representation=False, parameters_identity=None
    ):
        if as_representation:
            for value, identity in zip(
                values, self.order(parameters_identity=parameters_identity)
            ):
                self.parameters[identity].set_representation(value)
            return
        for value, identity in zip(
            values, self.order(parameters_identity=parameters_identity)
        ):
            self.parameters[identity].set_value(value)

    def set_uncertainty(
        self,
        uncertainty,
        as_representation=False,
        parameters_identity=None,
    ):
        uncertainty = torch.as_tensor(
            uncertainty, dtype=AP_config.ap_dtype, device=AP_config.ap_device
        )
        # track the order of the parameters
        porder = self.order(parameters_identity=parameters_identity)

        # If uncertainty is provided by identity, they are individually updated
        if parameters_identity is not None:
            parameters_identity = list(parameters_identity)
            for P in porder:
                for pid in self.get_id(P).identities:
                    if pid in parameters_identity:
                        self.get_id(P).set_uncertainty(
                            uncertainty[parameters_identity.index(pid)],
                            as_representation=as_representation,
                            identity=pid,
                        )
            return

        # If uncertainty is provided as the full vector, they are added in bulk
        start = 0
        for P, V in zip(
            porder,
            self.vector_len(),
        ):
            self.get_id(P).set_uncertainty(
                uncertainty[start : start + V].reshape(
                    self.get_id(P).representation.shape
                ),
                as_representation=as_representation,
            )
            start += V

    def iter_all(self):
        return self.parameters.values()

    def iter_top_level(self):
        return filter(
            lambda p: p.identity in self.top_level_parameters, self.parameters.values()
        )

    def __iter__(self):
        return filter(lambda p: not p.locked, self.parameters.values())

    def get_id(self, key):
        if ":" in key:
            return self.parameters[key[: key.find(":")]]
        else:
            return self.parameters[key]

    def get_name(self, key):
        # The : character is used for nested parameter group names
        if ":" in key:
            return self.groups[key[: key.find(":")]].get_name(key[key.find(":") + 1 :])

        # Attempt to find the parameter with that key name
        for Pid in self.top_level_parameters:
            if self.parameters[Pid].name == key:
                return self.parameters[Pid]
        # If the key cannot be found, raise an error
        raise KeyError()

    def pop_id(self, key):
        try:
            self.top_level_parameters.remove(key)
        except KeyError:
            pass
        return self.parameters.pop(key)

    def replace(self, old_param, new_param):
        was_top_level = old_param.identity in self.top_level_parameters
        self.pop_id(old_param.identity)
        self.add_parameter(new_param, top_level=was_top_level)

    def to(self, dtype=None, device=None):
        if dtype is None:
            dtype = AP_config.ap_dtype
        if device is None:
            device = AP_config.ap_device
        for P in self.parameters.values():
            P.to(dtype=dtype, device=device)
        return self

    def get_state(self, save_groups=True):
        state = {"name": self.name}
        if len(self.parameters) > 0:
            state["parameter_order"] = list(P.name for P in self.iter_top_level())
            state["parameters"] = dict(
                (P.name, P.get_state()) for P in self.iter_top_level()
            )
        if save_groups and len(self.groups) > 0:
            state["groups"] = dict(
                (G.name, G.get_state()) for G in self.groups.values()
            )
        return state

    def set_state(self, state):
        self.name = state["name"]
        if "parameters" in state:
            self.parameters = OrderedDict()
            for param in state["parameter_order"]:
                self.add_parameter(
                    Parameter(**state["parameters"][param]), top_level=True
                )
        if "groups" in state:
            self.groups = OrderedDict()
            for group in state["groups"]:
                self.add_group(Parameter_Group(group, state=state["groups"][group]))

    def __getitem__(self, key):
        return self.get_name(key)

    def __contains__(self, key):
        try:
            self.get_name(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        return len(self.parameters)

    def __str__(self):
        return f"Parameter Group: {self.name}\n" + "\n".join(
            list(str(P) for P in self.iter_all())
        )
