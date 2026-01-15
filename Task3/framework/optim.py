import numpy as np

from .tensor import tensor_from_numpy, tensor_to_numpy


class SGD:
    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            w = tensor_to_numpy(p.data)
            g = tensor_to_numpy(p.grad)
            if self.weight_decay > 0:
                g = g + self.weight_decay * w
            w = w - self.lr * g
            p.data = tensor_from_numpy(w, p.data.device())


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            w = tensor_to_numpy(p.data)
            g = tensor_to_numpy(p.grad)
            if self.weight_decay > 0:
                g = g + self.weight_decay * w
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(w)
                self.v[key] = np.zeros_like(w)
            self.m[key] = b1 * self.m[key] + (1 - b1) * g
            self.v[key] = b2 * self.v[key] + (1 - b2) * (g ** 2)
            m_hat = self.m[key] / (1 - b1 ** self.t)
            v_hat = self.v[key] / (1 - b2 ** self.t)
            w = w - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data = tensor_from_numpy(w, p.data.device())

