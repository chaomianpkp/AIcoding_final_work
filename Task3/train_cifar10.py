import argparse
import numpy as np
import torchvision
from torchvision import transforms

from framework.tensor import Tensor, Device, tensor_from_numpy
from framework.nn import SimpleCNN
from framework.ops import softmax_cross_entropy
from framework.optim import SGD, Adam
from framework.utils import set_seed, batch_iter, accuracy


def load_cifar10(root, train=True):
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    X = np.stack([np.array(dataset[i][0], dtype=np.float32) for i in range(len(dataset))], axis=0)
    y = np.array([dataset[i][1] for i in range(len(dataset))], dtype=np.int64)
    return X, y


def evaluate(model, X, y, batch_size=64, device=Device.CPU):
    preds_all = []
    for Xb, _ in batch_iter(X, y, batch_size=batch_size, shuffle=False):
        xb = Tensor(tensor_from_numpy(Xb, device), requires_grad=False)
        logits = model(xb)
        preds = np.argmax(logits.numpy(), axis=1)
        preds_all.append(preds)
    preds_all = np.concatenate(preds_all)
    return accuracy(preds_all, y)


def train(args):
    set_seed(args.seed)
    device = Device.GPU if args.device == "gpu" else Device.CPU

    X_train, y_train = load_cifar10(args.data_root, train=True)
    X_test, y_test = load_cifar10(args.data_root, train=False)

    model = SimpleCNN(num_classes=10, device=device)
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        losses = []
        batch_count = 0
        total_batches = (len(X_train) + args.batch_size - 1) // args.batch_size
        for Xb, yb in batch_iter(X_train, y_train, batch_size=args.batch_size, shuffle=True):
            batch_count += 1
            if batch_count % 50 == 0 or batch_count == 1:
                print(f"  Processing batch {batch_count}/{total_batches}...")
            xb = Tensor(tensor_from_numpy(Xb, device), requires_grad=False)
            yb = Tensor(tensor_from_numpy(yb.astype(np.float32), device), requires_grad=False)
            model.zero_grad()
            logits = model(xb)
            loss = softmax_cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.numpy()[0]))

        train_acc = evaluate(model, X_train[:5000], y_train[:5000], batch_size=args.batch_size, device=device)
        test_acc = evaluate(model, X_test, y_test, batch_size=args.batch_size, device=device)
        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")


def build_argparser():
    parser = argparse.ArgumentParser(description="Task3 CIFAR-10 training (custom CUDA framework)")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())

