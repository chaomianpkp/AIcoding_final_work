import numpy as np

from .tensor import Tensor, Device, tensor_from_numpy
from . import ops


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def parameters(self):
        params = []
        for _, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ReLU(Module):
    def forward(self, x):
        return ops.relu(x)


class MaxPool2d(Module):
    def forward(self, x):
        return ops.maxpool2d(x)


class Flatten(Module):
    def forward(self, x):
        return ops.flatten(x)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, device=Device.CPU):
        weight = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32) * 0.05
        bias = np.zeros((out_channels,), dtype=np.float32)
        self.weight = Parameter(tensor_from_numpy(weight, device))
        self.bias = Parameter(tensor_from_numpy(bias, device))

    def forward(self, x):
        return ops.conv2d(x, self.weight, self.bias)


class Linear(Module):
    def __init__(self, in_features, out_features, device=Device.CPU):
        weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.05
        bias = np.zeros((out_features,), dtype=np.float32)
        self.weight = Parameter(tensor_from_numpy(weight, device))
        self.bias = Parameter(tensor_from_numpy(bias, device))

    def forward(self, x):
        return ops.linear(x, self.weight, self.bias)


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params


class SimpleCNN(Module):
    def __init__(self, num_classes=10, device=Device.CPU):
        self.features = Sequential(
            Conv2d(3, 16, device=device),
            ReLU(),
            MaxPool2d(),
            Conv2d(16, 32, device=device),
            ReLU(),
            MaxPool2d(),
            Flatten(),
        )
        self.classifier = Linear(32 * 8 * 8, num_classes, device=device)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

