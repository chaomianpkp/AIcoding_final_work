"""
High level helpers for loading MNIST data as custom Tensor objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required to load MNIST data") from exc

from . import Device, tensor_from_numpy

if TYPE_CHECKING:
    from . import Tensor


def load_mnist_to_tensor(
    data_dir: str | Path = "~/.mnist",
    train: bool = True,
    limit: Optional[int] = None,
    device: Device = Device.CPU,
    normalize: bool = True,
) -> Tuple["Tensor", "Tensor"]:
    """
    Download (if necessary) MNIST and return (images, labels) as custom Tensors.
    """

    root = Path(data_dir).expanduser()
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=str(root), train=train, download=True, transform=transform)

    data = dataset.data.numpy().astype(np.float32)
    targets = dataset.targets.numpy().astype(np.int64)

    if limit is not None:
        data = data[:limit]
        targets = targets[:limit]

    if normalize:
        data /= 255.0

    data = data[:, None, :, :]  # add channel dimension -> [N, 1, 28, 28]
    labels = targets.astype(np.float32)  # our Tensor currently stores floats

    images_tensor = tensor_from_numpy(data, device=device)
    labels_tensor = tensor_from_numpy(labels, device=device)
    return images_tensor, labels_tensor

