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
        self._load_backend(backend)
        self._backend = backend

    def _load_backend(self, backend):
        if backend == "torch":
            self.module = importlib.import_module("torch")
            self.setup_torch()
        elif backend == "jax":
            self.module = importlib.import_module("jax.numpy")
            self.setup_jax()
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
        self.gammaln = self._gammaln_torch
        self.logit = self._logit_torch
        self.sigmoid = self._sigmoid_torch
        self.repeat = self._repeat_torch
        self.stack = self._stack_torch
        self.transpose = self._transpose_torch
        self.upsample2d = self._upsample2d_torch
        self.pad = self._pad_torch
        self.LinAlgErr = self.module._C._LinAlgError
        self.roll = self._roll_torch
        self.clamp = self._clamp_torch
        self.flatten = self._flatten_torch
        self.conv2d = self._conv2d_torch
        self.mean = self._mean_torch
        self.sum = self._sum_torch
        self.max = self._max_torch
        self.topk = self._topk_torch
        self.bessel_j1 = self._bessel_j1_torch
        self.bessel_k1 = self._bessel_k1_torch
        self.lgamma = self._lgamma_torch
        self.hessian = self._hessian_torch
        self.jacobian = self._jacobian_torch
        self.jacfwd = self._jacfwd_torch
        self.grad = self._grad_torch
        self.vmap = self._vmap_torch
        self.long = self._long_torch
        self.fill_at_indices = self._fill_at_indices_torch
        self.add_at_indices = self._add_at_indices_torch
        self.and_at_indices = self._and_at_indices_torch

    def setup_jax(self):
        self.jax = importlib.import_module("jax")
        self.jax.config.update("jax_enable_x64", True)
        config.DTYPE = None
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
        self.gammaln = self._gammaln_jax
        self.logit = self._logit_jax
        self.sigmoid = self._sigmoid_jax
        self.repeat = self._repeat_jax
        self.stack = self._stack_jax
        self.transpose = self._transpose_jax
        self.upsample2d = self._upsample2d_jax
        self.pad = self._pad_jax
        self.LinAlgErr = Exception
        self.roll = self._roll_jax
        self.clamp = self._clamp_jax
        self.flatten = self._flatten_jax
        self.conv2d = self._conv2d_jax
        self.mean = self._mean_jax
        self.sum = self._sum_jax
        self.max = self._max_jax
        self.topk = self._topk_jax
        self.bessel_j1 = self._bessel_j1_jax
        self.bessel_k1 = self._bessel_k1_jax
        self.lgamma = self._lgamma_jax
        self.hessian = self._hessian_jax
        self.jacobian = self._jacobian_jax
        self.jacfwd = self._jacfwd_jax
        self.grad = self._grad_jax
        self.vmap = self._vmap_jax
        self.long = self._long_jax
        self.fill_at_indices = self._fill_at_indices_jax
        self.add_at_indices = self._add_at_indices_jax
        self.and_at_indices = self._and_at_indices_jax

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

    def _concatenate_torch(self, arrays, dim=0):
        return self.module.cat(arrays, dim=dim)

    def _concatenate_jax(self, arrays, dim=0):
        return self.module.concatenate(arrays, axis=dim)

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

    def _repeat_torch(self, a, repeats, axis=None):
        return self.module.repeat_interleave(a, repeats, dim=axis)

    def _repeat_jax(self, a, repeats, axis=None):
        return self.module.repeat(a, repeats, axis=axis)

    def _stack_torch(self, arrays, dim=0):
        return self.module.stack(arrays, dim=dim)

    def _stack_jax(self, arrays, dim=0):
        return self.module.stack(arrays, axis=dim)

    def _transpose_torch(self, array, *args):
        return self.module.transpose(array, *args)

    def _transpose_jax(self, array, *args):
        permutation = np.arange(array.ndim)
        permutation[np.sort(args)] = args
        return self.module.transpose(array, permutation)

    def _gammaln_torch(self, array):
        return self.module.special.gammaln(array)

    def _gammaln_jax(self, array):
        return self.jax.scipy.special.gammaln(array)

    def _sigmoid_torch(self, array):
        return self.module.sigmoid(array)

    def _sigmoid_jax(self, array):
        return self.jax.nn.sigmoid(array)

    def _logit_torch(self, array):
        return self.module.logit(array)

    def _logit_jax(self, array):
        return self.jax.scipy.special.logit(array)

    def _upsample2d_torch(self, array, scale_factor, method):
        U = self.module.nn.Upsample(scale_factor=scale_factor, mode=method)
        array = U(array) / scale_factor**2
        return array

    def _upsample2d_jax(self, array, scale_factor, method):
        if method == "nearest":
            method = "bilinear"  # no nearest neighbor interpolation in jax
        new_shape = list(array.shape)
        new_shape[-2] = array.shape[-2] * scale_factor
        new_shape[-1] = array.shape[-1] * scale_factor
        return self.jax.image.resize(array, new_shape, method=method)

    def _pad_torch(self, array, padding, mode):
        return self.module.nn.functional.pad(array, padding[-4:], mode=mode)

    def _pad_jax(self, array, padding, mode):
        if mode == "replicate":
            mode = "edge"
        padding = np.array(padding).reshape(-1, 2)
        return self.module.pad(array, padding, mode=mode)

    def _roll_torch(self, array, shifts, dims):
        return self.module.roll(array, shifts, dims=dims)

    def _roll_jax(self, array, shifts, dims):
        return self.module.roll(array, shifts, axis=dims)

    def _clamp_torch(self, array, min, max):
        return self.module.clamp(array, min, max)

    def _clamp_jax(self, array, min, max):
        return self.module.clip(array, min, max)

    def _long_torch(self, array):
        return array.long()

    def _long_jax(self, array):
        return self.module.astype(array, self.module.int64)

    def _conv2d_torch(self, input, kernel, padding, stride=1):
        return self.module.nn.functional.conv2d(
            input,
            kernel,
            padding=padding,
            stride=stride,
        )

    def _conv2d_jax(self, input, kernel, padding, stride=1):
        return self.jax.lax.conv_general_dilated(
            input, kernel, window_strides=(stride, stride), padding=padding
        )

    def _mean_torch(self, array, dim=None):
        return self.module.mean(array, dim=dim)

    def _mean_jax(self, array, dim=None):
        return self.module.mean(array, axis=dim)

    def _sum_torch(self, array, dim=None):
        return self.module.sum(array, dim=dim)

    def _sum_jax(self, array, dim=None):
        return self.module.sum(array, axis=dim)

    def _max_torch(self, array, dim=None):
        return array.amax(dim=dim)

    def _max_jax(self, array, dim=None):
        return self.module.max(array, axis=dim)

    def _topk_torch(self, array, k):
        return self.module.topk(array, k=k)

    def _topk_jax(self, array, k):
        return self.jax.lax.top_k(array, k=k)

    def _bessel_j1_torch(self, array):
        return self.module.special.bessel_j1(array)

    def _bessel_j1_jax(self, array):
        return self.jax.scipy.special.bessel_jn(array, v=1)[-1]

    def _bessel_k1_torch(self, array):
        return self.module.special.modified_bessel_k1(array)

    def _bessel_k1_jax(self, array):
        return self.jax.scipy.special.kn(1, array)

    def _lgamma_torch(self, array):
        return self.module.lgamma(array)

    def _lgamma_jax(self, array):
        return self.jax.lax.lgamma(array)

    def _grad_torch(self, func):
        return self.module.func.grad(func)

    def _grad_jax(self, func):
        return self.jax.grad(func)

    def _jacobian_torch(self, func, x, strategy="forward-mode", vectorize=True, create_graph=False):
        return self.module.autograd.functional.jacobian(
            func, x, strategy=strategy, vectorize=vectorize, create_graph=create_graph
        )

    def _jacobian_jax(self, func, x, strategy="forward-mode", vectorize=True, create_graph=False):
        if "forward" in strategy:
            # n = x.size
            # eye = self.module.eye(n)
            # Jt = self.jax.vmap(lambda s: self.jax.jvp(func, (x,), (s,))[1])(eye)
            # return self.module.moveaxis(Jt, 0, -1)
            return self.jax.jacfwd(func)(x)
        return self.jax.jacrev(func)(x)

    def _jacfwd_torch(self, func):
        return self.module.func.jacfwd(func)

    def _jacfwd_jax(self, func):
        return self.jax.jacfwd(func)

    def _hessian_torch(self, func):
        return self.module.func.hessian(func)

    def _hessian_jax(self, func):
        return self.jax.hessian(func)

    def _vmap_torch(self, *args, **kwargs):
        return self.module.vmap(*args, **kwargs)

    def _vmap_jax(self, *args, **kwargs):
        return self.jax.vmap(*args, **kwargs)

    def _fill_at_indices_torch(self, array, indices, values):
        array[indices] = values
        return array

    def _fill_at_indices_jax(self, array, indices, values):
        array = array.at[indices].set(values)
        return array

    def _add_at_indices_torch(self, array, indices, values):
        array[indices] += values
        return array

    def _add_at_indices_jax(self, array, indices, values):
        array = array.at[indices].add(values)
        return array

    def _and_at_indices_torch(self, array, indices, values):
        array[indices] &= values
        return array

    def _and_at_indices_jax(self, array, indices, values):
        array = array.at[indices].set(array[indices] & values)
        return array

    def _flatten_torch(self, array, start_dim=0, end_dim=-1):
        return array.flatten(start_dim, end_dim)

    def _flatten_jax(self, array, start_dim=0, end_dim=-1):
        shape = tuple(array.shape)
        end_dim = (end_dim % len(shape)) + 1
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim:]
        return self.module.reshape(array, new_shape)

    def arange(self, *args, dtype=None, device=None):
        return self.module.arange(*args, dtype=dtype, device=device)

    def linspace(self, start, end, steps, dtype=None, device=None):
        return self.module.linspace(start, end, steps, dtype=dtype, device=device)

    def meshgrid(self, *arrays, indexing="ij"):
        return self.module.meshgrid(*arrays, indexing=indexing)

    def searchsorted(self, array, value):
        return self.module.searchsorted(array, value)

    def any(self, array):
        return self.module.any(array)

    def all(self, array):
        return self.module.all(array)

    def log(self, array):
        return self.module.log(array)

    def log10(self, array):
        return self.module.log10(array)

    def exp(self, array):
        return self.module.exp(array)

    def sin(self, array):
        return self.module.sin(array)

    def cos(self, array):
        return self.module.cos(array)

    def cosh(self, array):
        return self.module.cosh(array)

    def sqrt(self, array):
        return self.module.sqrt(array)

    def abs(self, array):
        return self.module.abs(array)

    def floor(self, array):
        return self.module.floor(array)

    def tanh(self, array):
        return self.module.tanh(array)

    def arctan(self, array):
        return self.module.arctan(array)

    def arctan2(self, y, x):
        return self.module.arctan2(y, x)

    def arcsin(self, array):
        return self.module.arcsin(array)

    def round(self, array):
        return self.module.round(array)

    def zeros(self, shape, dtype=None, device=None):
        return self.module.zeros(shape, dtype=dtype, device=device)

    def zeros_like(self, array, dtype=None):
        return self.module.zeros_like(array, dtype=dtype)

    def ones(self, shape, dtype=None, device=None):
        return self.module.ones(shape, dtype=dtype, device=device)

    def ones_like(self, array, dtype=None):
        return self.module.ones_like(array, dtype=dtype)

    def empty(self, shape, dtype=None, device=None):
        return self.module.empty(shape, dtype=dtype, device=device)

    def eye(self, n, dtype=None, device=None):
        return self.module.eye(n, dtype=dtype, device=device)

    def diag(self, array):
        return self.module.diag(array)

    def outer(self, a, b):
        return self.module.outer(a, b)

    def minimum(self, a, b):
        return self.module.minimum(a, b)

    def maximum(self, a, b):
        return self.module.maximum(a, b)

    def isnan(self, array):
        return self.module.isnan(array)

    def isfinite(self, array):
        return self.module.isfinite(array)

    def where(self, condition, x, y):
        return self.module.where(condition, x, y)

    def allclose(self, a, b, rtol=1e-5, atol=1e-8):
        return self.module.allclose(a, b, rtol=rtol, atol=atol)

    @property
    def linalg(self):
        return self.module.linalg

    @property
    def fft(self):
        return self.module.fft

    @property
    def inf(self):
        return self.module.inf

    @property
    def bool(self):
        return self.module.bool

    @property
    def int32(self):
        return self.module.int32

    @property
    def float32(self):
        return self.module.float32

    @property
    def float64(self):
        return self.module.float64


backend = Backend()
