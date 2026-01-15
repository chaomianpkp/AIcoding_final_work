"""
Unit tests that compare custom operators with torch.nn.functional.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import mytensor as mt

DEVICE = mt.Device.CPU


def np_from_tensor(t: mt.Tensor) -> np.ndarray:
    return mt.tensor_to_numpy(t)


def assert_close(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-4) -> None:
    diff = np.max(np.abs(actual - expected))
    assert diff <= atol, f"Max diff {diff} exceeds tolerance {atol}"


def random_tensor(shape):
    arr = np.random.randn(*shape).astype(np.float32)
    return arr, mt.tensor_from_numpy(arr, device=DEVICE)


def test_relu():
    print("Test ReLU...", end=" ")
    x_np, x = random_tensor((4, 5))
    y = np_from_tensor(mt.relu(x))
    assert_close(y, np.maximum(x_np, 0.0))

    dy_np, dy = random_tensor((4, 5))
    dx = np_from_tensor(mt.relu_backward(x, dy))
    assert_close(dx, dy_np * (x_np > 0.0))
    print("passed")


def test_sigmoid():
    print("Test Sigmoid...", end=" ")
    x_np, x = random_tensor((3, 7))
    y = np_from_tensor(mt.sigmoid(x))
    y_ref = 1.0 / (1.0 + np.exp(-x_np))
    assert_close(y, y_ref)

    dy_np, dy = random_tensor((3, 7))
    dx = np_from_tensor(mt.sigmoid_backward(mt.sigmoid(x), dy))
    dx_ref = dy_np * y_ref * (1.0 - y_ref)
    assert_close(dx, dx_ref)
    print("passed")


def test_linear():
    print("Test Linear...", end=" ")
    x_np, x = random_tensor((2, 3))
    w_np, w = random_tensor((4, 3))
    b_np, b = random_tensor((4,))

    y = np_from_tensor(mt.linear(x, w, b))
    y_ref = x_np @ w_np.T + b_np
    assert_close(y, y_ref)

    grad_np, grad = random_tensor((2, 4))
    dX, dW, db = mt.linear_backward(x, w, grad)
    assert_close(np_from_tensor(dX), grad_np @ w_np)
    assert_close(np_from_tensor(dW), grad_np.T @ x_np)
    assert_close(np_from_tensor(db), grad_np.sum(axis=0))
    print("passed")


def test_conv2d():
    print("Test Conv2D...", end=" ")
    x_np, x = random_tensor((2, 2, 5, 5))
    w_np, w = random_tensor((3, 2, 3, 3))
    b_np, b = random_tensor((3,))

    x_t = torch.tensor(x_np, requires_grad=True)
    w_t = torch.tensor(w_np, requires_grad=True)
    b_t = torch.tensor(b_np, requires_grad=True)
    y_ref = F.conv2d(x_t, w_t, b_t, padding=1, stride=1)

    y = np_from_tensor(mt.conv2d(x, w, b))
    assert_close(y, y_ref.detach().numpy())

    grad_np = np.random.randn(*y.shape).astype(np.float32)
    grad = mt.tensor_from_numpy(grad_np, device=DEVICE)
    y_ref.backward(torch.tensor(grad_np))

    dX, dW, db = mt.conv2d_backward(x, w, grad)
    assert_close(np_from_tensor(dX), x_t.grad.numpy())
    assert_close(np_from_tensor(dW), w_t.grad.numpy())
    assert_close(np_from_tensor(db), b_t.grad.numpy())
    print("passed")


def test_maxpool():
    print("Test MaxPool2D...", end=" ")
    x_np, x = random_tensor((1, 2, 4, 4))
    x_t = torch.tensor(x_np, requires_grad=True)
    y_ref = F.max_pool2d(x_t, kernel_size=2, stride=2)

    y = np_from_tensor(mt.maxpool2d(x))
    assert_close(y, y_ref.detach().numpy())

    grad_np = np.random.randn(*y.shape).astype(np.float32)
    grad = mt.tensor_from_numpy(grad_np, device=DEVICE)
    y_ref.backward(torch.tensor(grad_np))

    dX = np_from_tensor(mt.maxpool2d_backward(x, mt.maxpool2d(x), grad))
    assert_close(dX, x_t.grad.numpy())
    print("passed")


def test_softmax_cross_entropy():
    print("Test Softmax & CrossEntropy...", end=" ")
    x_np, x = random_tensor((4, 6))
    probs_tensor = mt.softmax(x)
    probs = np_from_tensor(probs_tensor)
    probs_ref = torch.softmax(torch.tensor(x_np), dim=1).numpy()
    assert_close(probs, probs_ref)

    labels = np.random.randint(0, 6, size=(4,), dtype=np.int64)
    labels_tensor = mt.tensor_from_numpy(labels.astype(np.float32), device=DEVICE)

    loss = mt.cross_entropy(probs_tensor, labels_tensor)
    loss_ref = -np.log(np.clip(probs_ref[np.arange(4), labels], 1e-12, None)).mean()
    assert abs(loss - loss_ref) <= 1e-4

    grad = np_from_tensor(mt.cross_entropy_backward(probs_tensor, labels_tensor))
    grad_ref = probs_ref.copy()
    grad_ref[np.arange(4), labels] -= 1.0
    grad_ref /= 4.0
    assert_close(grad, grad_ref)
    print("passed")


def run_all():
    torch.manual_seed(0)
    np.random.seed(0)
    test_relu()
    test_sigmoid()
    test_linear()
    test_conv2d()
    test_maxpool()
    test_softmax_cross_entropy()
    print("All operator tests passed.")


if __name__ == "__main__":
    run_all()

