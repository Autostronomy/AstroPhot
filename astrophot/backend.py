import os
import importlib
from typing import Annotated

from torch import Tensor, dtype, device
import numpy as np
import torch
from . import config

ArrayLike = Annotated[
    Tensor,
    "One of: torch.Tensor or jax.numpy.ndarray depending on the chosen backend.",
]
dtypeLike = Annotated[
    dtype,
    "One of: torch.dtype or jax.numpy.dtype depending on the chosen backend.",
]
deviceLike = Annotated[
    device,
    "One of: torch.device or jax.DeviceArray depending on the chosen backend.",
]


class Backend:
    def __init__(self, backend=None):
        self.backend = backend

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend):
        if backend is None:
            backend = os.getenv("CASKADE_BACKEND", "torch")
        self.module = self._load_backend(backend)
        self._backend = backend

    def _load_backend(self, backend):
        if backend == "torch":
            self.setup_torch()
            return importlib.import_module("torch")
        elif backend == "jax":
            self.setup_jax()
            return importlib.import_module("jax.numpy")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def setup_torch(self):
        config.DTYPE = torch.float64
        config.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.make_array = self._make_array_torch
        self._array_type = self._array_type_torch
        self.concatenate = self._concatenate_torch
        self.copy = self._copy_torch
        self.tolist = self._tolist_torch
        self.view = self._view_torch
        self.as_array = self._as_array_torch
        self.to = self._to_torch
        self.to_numpy = self._to_numpy_torch
        self.logit = self._logit_torch
        self.sigmoid = self._sigmoid_torch
        self.arange = self._arange_torch
        self.meshgrid = self._meshgrid_torch
        self.repeat = self._repeat_torch
        self.stack = self._stack_torch
        self.transpose = self._transpose_torch

    def setup_jax(self):
        self.jax = importlib.import_module("jax")
        self.jax.config.update("jax_enable_x64", True)
        config.DTYPE = self.jax.numpy.float64
        config.DEVICE = None
        self.make_array = self._make_array_jax
        self._array_type = self._array_type_jax
        self.concatenate = self._concatenate_jax
        self.copy = self._copy_jax
        self.tolist = self._tolist_jax
        self.view = self._view_jax
        self.as_array = self._as_array_jax
        self.to = self._to_jax
        self.to_numpy = self._to_numpy_jax
        self.logit = self._logit_jax
        self.sigmoid = self._sigmoid_jax
        self.arange = self._arange_jax
        self.meshgrid = self._meshgrid_jax
        self.repeat = self._repeat_jax
        self.stack = self._stack_jax
        self.transpose = self._transpose_jax

    @property
    def array_type(self):
        return self._array_type()

    def _make_array_torch(self, array, dtype=None, device=None):
        return self.module.tensor(array, dtype=dtype, device=device)

    def _make_array_jax(self, array, dtype=None, **kwargs):
        return self.module.array(array, dtype=dtype)

    def _array_type_torch(self):
        return self.module.Tensor

    def _array_type_jax(self):
        return self.module.ndarray

    def _concatenate_torch(self, arrays, axis=0):
        return self.module.cat(arrays, dim=axis)

    def _concatenate_jax(self, arrays, axis=0):
        return self.module.concatenate(arrays, axis=axis)

    def _copy_torch(self, array):
        return array.detach().clone()

    def _copy_jax(self, array):
        return self.module.copy(array)

    def _tolist_torch(self, array):
        return array.detach().cpu().tolist()

    def _tolist_jax(self, array):
        return array.block_until_ready().tolist()

    def _view_torch(self, array, shape):
        return array.reshape(shape)

    def _view_jax(self, array, shape):
        return array.reshape(shape)

    def _as_array_torch(self, array, dtype=None, device=None):
        return self.module.as_tensor(array, dtype=dtype, device=device)

    def _as_array_jax(self, array, dtype=None, **kwargs):
        return self.module.asarray(array, dtype=dtype)

    def _to_torch(self, array, dtype=None, device=None):
        return array.to(dtype=dtype, device=device)

    def _to_jax(self, array, dtype=None, device=None):
        return self.jax.device_put(array.astype(dtype), device=device)

    def _to_numpy_torch(self, array):
        return array.detach().cpu().numpy()

    def _to_numpy_jax(self, array):
        return np.array(array.block_until_ready())

    def _arange_torch(self, *args, dtype=None, device=None):
        return self.module.arange(*args, dtype=dtype, device=device)

    def _arange_jax(self, *args, dtype=None, device=None):
        return self.jax.arange(*args, dtype=dtype, device=device)

    def _meshgrid_torch(self, *arrays, indexing="ij"):
        return self.module.meshgrid(*arrays, indexing=indexing)

    def _meshgrid_jax(self, *arrays, indexing="ij"):
        return self.jax.meshgrid(*arrays, indexing=indexing)

    def _repeat_torch(self, a, repeats, axis=None):
        return self.module.repeat_interleave(a, repeats, dim=axis)

    def _repeat_jax(self, a, repeats, axis=None):
        return self.jax.repeat(a, repeats, axis=axis)

    def _stack_torch(self, arrays, dim=0):
        return self.module.stack(arrays, dim=dim)

    def _stack_jax(self, arrays, dim=0):
        return self.jax.stack(arrays, axis=dim)

    def _transpose_torch(self, array, *args):
        return self.module.transpose(array, *args)

    def _transpose_jax(self, array, *args):
        return self.jax.transpose(array, args)

    def _sigmoid_torch(self, array):
        return self.module.sigmoid(array)

    def _sigmoid_jax(self, array):
        return self.jax.nn.sigmoid(array)

    def _logit_torch(self, array):
        return self.module.logit(array)

    def _logit_jax(self, array):
        return self.jax.scipy.special.logit(array)

    def _clone_torch(self, array):
        return array.clone()

    def _clone_jax(self, array):
        return self.module.copy(array)

    def any(self, array):
        return self.module.any(array)

    def all(self, array):
        return self.module.all(array)

    def log(self, array):
        return self.module.log(array)

    def exp(self, array):
        return self.module.exp(array)

    def sin(self, array):
        return self.module.sin(array)

    def cos(self, array):
        return self.module.cos(array)

    def sqrt(self, array):
        return self.module.sqrt(array)

    def arctan(self, array):
        return self.module.arctan(array)

    def arctan2(self, y, x):
        return self.module.arctan2(y, x)

    def arcsin(self, array):
        return self.module.arcsin(array)

    def sum(self, array, axis=None):
        return self.module.sum(array, axis=axis)

    def zeros(self, shape, dtype=None, device=None):
        return self.module.zeros(shape, dtype=dtype, device=device)

    def zeros_like(self, array):
        return self.module.zeros_like(array)

    def ones(self, shape, dtype=None, device=None):
        return self.module.ones(shape, dtype=dtype, device=device)

    def ones_like(self, array):
        return self.module.ones_like(array)

    def empty(self, shape, dtype=None, device=None):
        return self.module.empty(shape, dtype=dtype, device=device)

    def minimum(self, a, b):
        return self.module.minimum(a, b)

    def maximum(self, a, b):
        return self.module.maximum(a, b)

    def isnan(self, array):
        return self.module.isnan(array)

    def where(self, condition, x, y):
        return self.module.where(condition, x, y)

    @property
    def linalg(self):
        return self.module.linalg

    @property
    def inf(self):
        return self.module.inf

    @property
    def bool(self):
        return self.module.bool

    @property
    def int32(self):
        return self.module.int32


backend = Backend()
