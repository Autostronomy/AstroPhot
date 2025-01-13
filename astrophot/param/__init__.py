from typing import Union

from caskade import Param, ActiveStateError
import torch
from torch import Tensor


class APParam(Param):

    def __init__(self, *args, uncertainty=None, default_value=None, locked=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.uncertainty = uncertainty
        self.default_value = default_value
        self.locked = locked

    @property
    def uncertainty(self):
        if self._uncertainty is None:
            try:
                return torch.zeros_like(self.value)
            except TypeError:
                pass
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            self._uncertainty = torch.as_tensor(value)
        else:
            self._uncertainty = None

    @property
    def default_value(self):
        return self._default_value

    @default_value.setter
    def default_value(self, value):
        if value is not None:
            self._default_value = torch.as_tensor(value)
        else:
            self._default_value = None

    @property
    def value(self) -> Union[Tensor, None]:
        if self.pointer and self._value is None:
            if self.active:
                self._value = self._pointer_func(self)
            else:
                return self._pointer_func(self)

        if self._value is None:
            return self._default_value
        return self._value

    @property
    def locked(self):
        return self._locked

    @locked.setter
    def locked(
        self, value
    ):  # fixme still working on the logic here. Static should always be locked, but dynamic may go either way, I think?
        self._locked = value
        if self._locked and self._value is None and self._default_value is not None:
            self.value = self.default_value
        if not self._locked and self._value is not None:
            self.default_value = self._value

    @value.setter
    def value(self, value):
        # While active no value can be set
        if self.active:
            raise ActiveStateError(f"Cannot set value of parameter {self.name} while active")

        # unlink if pointer to avoid floating references
        if self.pointer:
            for child in tuple(self.children.values()):
                self.unlink(child)

        if value is None:
            self._type = "dynamic"
            self._pointer_func = None
            self._value = None
        elif isinstance(value, Param):
            self._type = "pointer"
            self.link(str(id(value)), value)
            self._pointer_func = lambda p: p[str(id(value))].value
            self._shape = None
            self._value = None
        elif callable(value):
            self._type = "pointer"
            self._shape = None
            self._pointer_func = value
            self._value = None
        elif self.locked:
            self._type = "static"
            value = torch.as_tensor(value)
            self.shape = value.shape
            self._value = value
            try:
                self.valid = self._valid  # re-check valid range
            except AttributeError:
                pass
        else:
            self._type = "dynamic"
            self._pointer_func = None
            self._value = None
            if value is not None:
                self.default_value = value

        self.update_graph()
