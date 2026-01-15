import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def accuracy(preds, labels):
    return float((preds == labels).mean())


def batch_iter(X, y, batch_size=64, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]

