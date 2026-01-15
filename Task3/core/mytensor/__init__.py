"""
Python helper APIs wrapping the compiled pybind11 extension.
"""

from __future__ import annotations

import numpy as np

try:
    from . import _C  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - readable error path
    raise ImportError(
        "mytensor extension is not built yet. Please run "
        "`python setup.py build_ext --inplace` inside the Task3/core directory first."
    ) from exc

Tensor = _C.Tensor
Device = _C.Device

# Tensor <-> numpy bridges ----------------------------------------------------

def tensor_from_numpy(array: np.ndarray, device: Device = Device.CPU) -> Tensor:
    arr = np.asarray(array, dtype=np.float32, order="C")
    return _C.tensor_from_numpy(arr, device)


def tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return np.array(_C.tensor_to_numpy(tensor), copy=True)


# Activation helpers ----------------------------------------------------------

def relu(x: Tensor) -> Tensor:
    return _C.relu_forward(x)


def relu_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    return _C.relu_backward(x, grad_output)


def sigmoid(x: Tensor) -> Tensor:
    return _C.sigmoid_forward(x)


def sigmoid_backward(y: Tensor, grad_output: Tensor) -> Tensor:
    return _C.sigmoid_backward(y, grad_output)


# Linear layer ----------------------------------------------------------------

def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return _C.linear_forward(x, weight, bias)


def linear_backward(x: Tensor, weight: Tensor, grad_output: Tensor):
    return _C.linear_backward(x, weight, grad_output)


# Convolution -----------------------------------------------------------------

def conv2d(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return _C.conv2d_forward(x, weight, bias)


def conv2d_backward(x: Tensor, weight: Tensor, grad_output: Tensor):
    return _C.conv2d_backward(x, weight, grad_output)


# Pooling ---------------------------------------------------------------------

def maxpool2d(x: Tensor) -> Tensor:
    return _C.maxpool2d_forward(x)


def maxpool2d_backward(x: Tensor, y: Tensor, grad_output: Tensor) -> Tensor:
    return _C.maxpool2d_backward(x, y, grad_output)


# Softmax + Cross Entropy -----------------------------------------------------

def softmax(x: Tensor) -> Tensor:
    return _C.softmax_forward(x)


def cross_entropy(probs: Tensor, labels: Tensor) -> float:
    return _C.cross_entropy_forward(probs, labels)


def cross_entropy_backward(probs: Tensor, labels: Tensor) -> Tensor:
    return _C.cross_entropy_backward(probs, labels)


__all__ = [
    "Tensor",
    "Device",
    "tensor_from_numpy",
    "tensor_to_numpy",
    "relu",
    "relu_backward",
    "sigmoid",
    "sigmoid_backward",
    "linear",
    "linear_backward",
    "conv2d",
    "conv2d_backward",
    "maxpool2d",
    "maxpool2d_backward",
    "softmax",
    "cross_entropy",
    "cross_entropy_backward",
]

