# Task2: PyTorch Data Parallel Implementation
# 目标：使用数据并行（DataParallel）来加速CIFAR-10训练

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    """改进的CNN网络结构"""
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
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
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 32x32 -> 16x16
        
        # 第二组：64 -> 128
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)  # 16x16 -> 8x8
        
        # 第三组：128 -> 256
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.max_pool2d(x, 2)  # 8x8 -> 4x4
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def get_data_loaders(data_root, batch_size, num_workers=2):
    """获取数据加载器"""
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, epochs, lr, use_parallel=False):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    best_test_acc = 0.0
    train_times = []
    
    print(f"\n{'='*60}")
    print(f"Training with {'DataParallel' if use_parallel else 'Single GPU/CPU'}")
    print(f"Device: {device}")
    if use_parallel and torch.cuda.is_available():
        print(f"Using {torch.cuda.device_count()} GPU(s)")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_time = time.time() - epoch_start
        train_times.append(epoch_time)
        
        # 更新学习率
        scheduler.step()
        
        # 测试阶段
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        train_acc = 100 * correct_train / total_train
        test_acc = 100 * correct_test / total_test
        avg_loss = running_loss / len(train_loader)
        
        print(f'Epoch [{epoch}/{epochs}] - Time: {epoch_time:.2f}s - '
              f'Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - '
              f'Test Acc: {test_acc:.2f}%')
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'  ✓ New best test accuracy: {best_test_acc:.2f}%')
        print()
    
    avg_time_per_epoch = sum(train_times) / len(train_times)
    total_time = sum(train_times)
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Average time per epoch: {avg_time_per_epoch:.2f}s")
    print(f"Total training time: {total_time:.2f}s")
    print(f"{'='*60}\n")
    
    return best_test_acc, avg_time_per_epoch, total_time


def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 检查GPU
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {device_count}")
    
    # 选择设备
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        print("Warning: Using CPU (slower)")
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    
    # 创建模型
    model = Net(num_classes=10)
    model = model.to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # 使用数据并行（如果有多GPU且启用）
    use_parallel = args.use_parallel and device_count > 1
    
    if use_parallel:
        # 使用DataParallel包装模型
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {device_count} GPUs")
        # 注意：DataParallel会自动将batch_size分成多份，每份给一个GPU
    else:
        if args.use_parallel and device_count <= 1:
            print("Warning: --use-parallel specified but only 1 GPU available, using single GPU")
    
    # 训练模型
    best_acc, avg_time, total_time = train_model(
        model, train_loader, test_loader, device, 
        args.epochs, args.lr, use_parallel
    )
    
    return {
        'best_accuracy': best_acc,
        'avg_time_per_epoch': avg_time,
        'total_time': total_time,
        'use_parallel': use_parallel,
        'num_gpus': device_count if use_parallel else 1
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Task2: PyTorch Data Parallel Training")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu",
                       help="Device: 'cpu' or 'gpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--use-parallel", action="store_true",
                       help="Use DataParallel for multi-GPU training")
    return parser


if __name__ == "__main__":
    results = main(build_argparser().parse_args())

