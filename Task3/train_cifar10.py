import argparse
import json
import os
import time
from datetime import datetime
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
    # 修复 numpy 2.0 兼容性问题
    X_list = []
    y_list = []
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]
        # 转换为 numpy array
        img_array = np.array(img_tensor, dtype=np.float32)
        X_list.append(img_array)
        y_list.append(label)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
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
    
    # 计算模型参数数量
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        # shape() 是方法，需要调用
        shape = p.data.shape()
        param_count = int(np.prod(shape))
        total_params += param_count
        trainable_params += param_count
    
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 初始化日志记录
    training_log = {
        'config': {
            'model': 'SimpleCNN',
            'framework': 'custom_cuda',
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': args.optimizer,
            'weight_decay': args.weight_decay,
            'device': str(device),
            'seed': args.seed,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
        },
        'training_history': [],
        'timing': {}
    }
    
    print(f"Model parameters: {total_params:,}")
    print(f"Device: {device}")
    print(f"Optimizer: {args.optimizer}, lr={args.lr}")
    print("\nStarting training...\n")
    
    best_test_acc = 0.0
    total_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
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
        
        train_time = time.time() - epoch_start_time
        
        # 评估
        test_start_time = time.time()
        train_acc = evaluate(model, X_train[:5000], y_train[:5000], batch_size=args.batch_size, device=device)
        test_acc = evaluate(model, X_test, y_test, batch_size=args.batch_size, device=device)
        test_time = time.time() - test_start_time
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        # 获取当前学习率（对于Adam，学习率是固定的）
        current_lr = args.lr
        
        # 记录日志
        epoch_log = {
            'epoch': epoch,
            'train_loss': float(avg_loss),
            'train_acc': float(train_acc * 100),  # 转换为百分比
            'test_acc': float(test_acc * 100),    # 转换为百分比
            'learning_rate': float(current_lr),
            'epoch_time': float(epoch_time),
            'test_time': float(test_time)
        }
        training_log['training_history'].append(epoch_log)
        
        print(f"Epoch {epoch}/{args.epochs} - Time: {epoch_time:.2f}s - "
              f"Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"  ✓ New best test accuracy: {best_test_acc*100:.2f}%\n")
        else:
            print()
    
    total_time = time.time() - total_start_time
    
    # 记录总时间信息
    training_log['timing'] = {
        'total_time': float(total_time),
        'avg_time_per_epoch': float(total_time / args.epochs),
        'best_test_acc': float(best_test_acc * 100)  # 转换为百分比
    }
    
    # 保存日志到文件
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含配置信息）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_name = args.optimizer
    device_name = "gpu" if args.device == "gpu" else "cpu"
    log_filename = f"training_log_task3_{optimizer_name}_{device_name}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'{"="*60}')
    print(f'Best test accuracy: {best_test_acc*100:.2f}%')
    print(f'Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)')
    print(f'Average time per epoch: {total_time/args.epochs:.2f}s')
    print(f'Training log saved to: {log_path}')
    print(f'{"="*60}\n')
    
    return training_log


def build_argparser():
    parser = argparse.ArgumentParser(description="Task3 CIFAR-10 training (custom CUDA framework)")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam",
                       help="Optimizer: 'sgd' or 'adam'")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                       help="Device: 'cpu' or 'gpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory to save training logs (JSON format)")
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())

