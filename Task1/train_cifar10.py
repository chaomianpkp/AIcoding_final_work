import argparse
import os
import json
import time
from datetime import datetime
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


class Net(nn.Module):
    """改进的CNN网络"""
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        # 第一组卷积：3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 第二组卷积：64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # 第三组卷积：128 -> 256
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # 全连接层
        # 经过3次MaxPool2d后，32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 初始化权重（Kaiming初始化）
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Kaiming初始化提升训练效果"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一组：3 -> 64
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16
        
        # 第二组：64 -> 128
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8
        
        # 第三组：128 -> 256
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)  # 8x8 -> 4x4
        
        # 展平
        x = torch.flatten(x, 1)  # [batch, 256*4*4]
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 训练时随机丢弃50%的神经元
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class SimpleNet(nn.Module):
    """原始的简单网络结构（用于对比）"""
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        x = self.fc3(f6)
        return x


def get_transforms(use_augmentation=False, train=True):
    """获取数据预处理转换"""
    if train and use_augmentation:
        # 训练时使用数据增强
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
            transforms.RandomCrop(32, padding=4),     # 随机裁剪（填充4像素）
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10均值标准差
        ])
    else:
        # 测试时不使用数据增强，但使用归一化
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def train(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 选择设备
    use_mlu = False
    try:
        use_mlu = torch.mlu.is_available()
    except:
        use_mlu = False

    if use_mlu:
        device = torch.device('mlu:0')
    else:
        if args.device == "gpu" and torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    
    # 定义数据集和数据加载器
    train_transform = get_transforms(use_augmentation=args.use_augmentation, train=True)
    test_transform = get_transforms(use_augmentation=False, train=False)
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    # 创建模型
    if args.model == "improved":
        model = Net(num_classes=10)
        print("Using Improved CNN (6 convolutional layers)")
    else:
        model = SimpleNet(num_classes=10)
        print("Using Simple CNN (original architecture)")
    
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        print(f"Using Adam optimizer, lr={args.lr}, weight_decay={args.weight_decay}")
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
        print(f"Using SGD optimizer, lr={args.lr}, momentum=0.9, weight_decay={args.weight_decay}")
    
    # 学习率调度器
    scheduler = None
    if args.use_lr_scheduler:
        if args.lr_scheduler_mode == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.lr_step_size, gamma=args.lr_decay_factor
            )
            print(f"Using StepLR scheduler: step_size={args.lr_step_size}, gamma={args.lr_decay_factor}")
        else:  # exp
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=args.lr_decay_factor
            )
            print(f"Using ExponentialLR scheduler: gamma={args.lr_decay_factor}")
    
    if args.use_augmentation:
        print("Data augmentation enabled: RandomHorizontalFlip + RandomCrop + Normalization")
    else:
        print("Data augmentation disabled")
    
    # 训练模型
    best_test_acc = 0.0
    
    # 初始化日志记录
    training_log = {
        'config': {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': args.optimizer,
            'weight_decay': args.weight_decay,
            'use_augmentation': args.use_augmentation,
            'use_lr_scheduler': args.use_lr_scheduler,
            'lr_scheduler_mode': args.lr_scheduler_mode if args.use_lr_scheduler else None,
            'lr_decay_factor': args.lr_decay_factor if args.use_lr_scheduler else None,
            'device': str(device),
            'seed': args.seed,
            'total_params': total_params,
            'trainable_params': trainable_params,
        },
        'training_history': [],
        'timing': {}
    }
    
    print("\nStarting training...\n")
    total_start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # 训练模式
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # 打印训练信息
            if (i + 1) % 100 == 0:
                train_acc = 100 * correct_train / total_train
                avg_loss = running_loss / (i + 1)
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            old_lr = current_lr
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            current_lr = new_lr
            if old_lr != new_lr:
                print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        
        # 测试模式
        test_start_time = time.time()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_time = time.time() - test_start_time
        epoch_time = time.time() - epoch_start_time
        
        test_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # 记录日志
        epoch_log = {
            'epoch': epoch,
            'train_loss': float(avg_loss),
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'learning_rate': float(current_lr),
            'epoch_time': float(epoch_time),
            'test_time': float(test_time)
        }
        training_log['training_history'].append(epoch_log)
        
        print(f'Epoch [{epoch}/{args.epochs}] - Time: {epoch_time:.2f}s - Loss: {avg_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'  ✓ New best test accuracy: {best_test_acc:.2f}%\n')
        else:
            print()
    
    total_time = time.time() - total_start_time
    
    # 记录总时间信息
    training_log['timing'] = {
        'total_time': float(total_time),
        'avg_time_per_epoch': float(total_time / args.epochs),
        'best_test_acc': float(best_test_acc)
    }
    
    # 保存日志到文件
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含配置信息）
    model_name = args.model
    optimizer_name = args.optimizer
    log_filename = f"training_log_{model_name}_{optimizer_name}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'{"="*60}')
    print(f'Best test accuracy: {best_test_acc:.2f}%')
    print(f'Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)')
    print(f'Average time per epoch: {total_time/args.epochs:.2f}s')
    print(f'Training log saved to: {log_path}')
    print(f'{"="*60}\n')
    
    return training_log


def build_argparser():
    parser = argparse.ArgumentParser(description="Task1 CIFAR-10 training (PyTorch)")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, 
                       help="Weight decay (L2 regularization)")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam",
                       help="Optimizer: 'sgd' or 'adam'")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu",
                       help="Device: 'cpu' or 'gpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, choices=["simple", "improved"], default="improved",
                       help="Model architecture: 'simple' (original) or 'improved' (deeper)")
    parser.add_argument("--use-augmentation", action="store_true",
                       help="Enable data augmentation (random flip + crop + normalization)")
    parser.add_argument("--use-lr-scheduler", action="store_true",
                       help="Enable learning rate scheduling")
    parser.add_argument("--lr-scheduler-mode", type=str, choices=["step", "exp"], default="exp",
                       help="Learning rate scheduler mode: 'step' or 'exp'")
    parser.add_argument("--lr-decay-factor", type=float, default=0.95,
                       help="Learning rate decay factor")
    parser.add_argument("--lr-step-size", type=int, default=5,
                       help="Step size for step scheduler (decay every N epochs)")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory to save training logs (JSON format)")
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
