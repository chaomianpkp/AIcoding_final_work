import numpy as np

from . import _C
from .tensor import tensor_from_numpy, tensor_to_numpy, Tensor


class Op:
    def backward(self, grad_output):
        raise NotImplementedError


class ReLUOp(Op):
    def __init__(self, x):
        self.x = x

    def forward(self):
        y = _C.relu_forward(self.x.data)
        return Tensor(y, requires_grad=self.x.requires_grad, op=self, inputs=[self.x])

    def backward(self, grad_output):
        dx = _C.relu_backward(self.x.data, grad_output)
        return [dx]


class LinearOp(Op):
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        y = _C.linear_forward(self.x.data, self.w.data, self.b.data)
        requires = self.x.requires_grad or self.w.requires_grad or self.b.requires_grad
        return Tensor(y, requires_grad=requires, op=self, inputs=[self.x, self.w, self.b])

    def backward(self, grad_output):
        dx, dw, db = _C.linear_backward(self.x.data, self.w.data, grad_output)
        return [dx, dw, db]


class Conv2dOp(Op):
    def __init__(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        y = _C.conv2d_forward(self.x.data, self.w.data, self.b.data)
        requires = self.x.requires_grad or self.w.requires_grad or self.b.requires_grad
        return Tensor(y, requires_grad=requires, op=self, inputs=[self.x, self.w, self.b])

    def backward(self, grad_output):
        dx, dw, db = _C.conv2d_backward(self.x.data, self.w.data, grad_output)
        return [dx, dw, db]


class MaxPool2dOp(Op):
    def __init__(self, x):
        self.x = x
        self.y = None

    def forward(self):
        y = _C.maxpool2d_forward(self.x.data)
        self.y = y
        return Tensor(y, requires_grad=self.x.requires_grad, op=self, inputs=[self.x])

    def backward(self, grad_output):
        dx = _C.maxpool2d_backward(self.x.data, self.y, grad_output)
        return [dx]


class FlattenOp(Op):
    def __init__(self, x):
        self.x = x
        self.orig_shape = None

    def forward(self):
        x_np = tensor_to_numpy(self.x.data)
        self.orig_shape = x_np.shape
        flat = x_np.reshape(x_np.shape[0], -1)
        y = tensor_from_numpy(flat, self.x.data.device())
        return Tensor(y, requires_grad=self.x.requires_grad, op=self, inputs=[self.x])

    def backward(self, grad_output):
        g_np = tensor_to_numpy(grad_output)
        g_np = g_np.reshape(self.orig_shape)
        dx = tensor_from_numpy(g_np, self.x.data.device())
        return [dx]


class SoftmaxCrossEntropyOp(Op):
    def __init__(self, logits, labels):
        self.logits = logits
        self.labels = labels
        self.probs = None

    def forward(self):
        self.probs = _C.softmax_forward(self.logits.data)
        loss_val = _C.cross_entropy_forward(self.probs, self.labels.data)
        loss_tensor = tensor_from_numpy(np.array([loss_val], dtype=np.float32), self.logits.data.device())
        requires = self.logits.requires_grad
        return Tensor(loss_tensor, requires_grad=requires, op=self, inputs=[self.logits, self.labels])

    def backward(self, grad_output):
        dlogits = _C.cross_entropy_backward(self.probs, self.labels.data)
        if grad_output is not None:
            scale = tensor_to_numpy(grad_output)[0]
            dlogits = tensor_from_numpy(tensor_to_numpy(dlogits) * scale, dlogits.device())
        return [dlogits, None]


def relu(x: Tensor):
    return ReLUOp(x).forward()


def linear(x: Tensor, w: Tensor, b: Tensor):
    return LinearOp(x, w, b).forward()


def conv2d(x: Tensor, w: Tensor, b: Tensor):
    return Conv2dOp(x, w, b).forward()


def maxpool2d(x: Tensor):
    return MaxPool2dOp(x).forward()


def flatten(x: Tensor):
    return FlattenOp(x).forward()


def softmax_cross_entropy(logits: Tensor, labels: Tensor):
    return SoftmaxCrossEntropyOp(logits, labels).forward()

