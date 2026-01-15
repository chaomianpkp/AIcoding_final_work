import numpy as np

from . import _C


Device = _C.Device


def tensor_from_numpy(array: np.ndarray, device: Device = Device.CPU):
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return _C.tensor_from_numpy(array, device)


def tensor_to_numpy(tensor):
    return _C.tensor_to_numpy(tensor)


def zeros_like(tensor):
    return _C.Tensor(list(tensor.shape()), tensor.device())


def add_tensors(a, b):
    a_np = tensor_to_numpy(a)
    b_np = tensor_to_numpy(b)
    return tensor_from_numpy(a_np + b_np, a.device())


class Tensor:
    def __init__(self, data, requires_grad=False, op=None, inputs=None):
        self.data = data
        self.requires_grad = requires_grad
        self.op = op
        self.inputs = inputs or []
        self.grad = None

    def numpy(self):
        return tensor_to_numpy(self.data)

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            grad = tensor_from_numpy(np.ones(self.data.shape(), dtype=np.float32), self.data.device())

        topo = []
        visited = set()

        def dfs(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            for inp in node.inputs:
                dfs(inp)
            topo.append(node)

        dfs(self)
        self.grad = grad

        for node in reversed(topo):
            if node.op is None:
                continue
            grads = node.op.backward(node.grad)
            for inp, g in zip(node.inputs, grads):
                if g is None:
                    continue
                if inp.grad is None:
                    inp.grad = g
                else:
                    inp.grad = add_tensors(inp.grad, g)


