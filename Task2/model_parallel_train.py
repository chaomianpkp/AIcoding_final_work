# Task2: PyTorch Model Parallel Implementation
# 目标：使用模型并行（Model Parallel）将模型的不同部分分布在不同GPU上

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ModelParallelNet(nn.Module):
    """模型并行网络：将模型的不同部分放在不同的GPU上"""
    def __init__(self, num_classes=10, device_ids=[0, 1]):
        super(ModelParallelNet, self).__init__()
        
        self.device_ids = device_ids if len(device_ids) >= 2 else [0, 0]
        self.device0 = torch.device(f'cuda:{self.device_ids[0]}')
        self.device1 = torch.device(f'cuda:{self.device_ids[1]}') if len(device_ids) >= 2 else self.device0
        
        # 第一部分的卷积层放在device0
        self.features_part1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        ).to(self.device0)
        
        # 第二部分的卷积层放在device1
        self.features_part2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        ).to(self.device1)
        
        # 全连接层放在device1
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        ).to(self.device1)
        
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
        # 输入数据在device0
        x = x.to(self.device0)
        
        # 第一部分在device0
        x = self.features_part1(x)
        
        # 将中间结果转移到device1
        x = x.to(self.device1)
        
        # 第二部分和分类器在device1
        x = self.features_part2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class SingleDeviceNet(nn.Module):
    """单设备网络（用于对比）"""
    def __init__(self, num_classes=10):
        super(SingleDeviceNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_data_loaders(data_root, batch_size, num_workers=2):
    """获取数据加载器"""
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


def train_model(model, train_loader, test_loader, device, epochs, lr, use_model_parallel=False):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    
    # 优化器需要包含所有设备的参数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    best_test_acc = 0.0
    train_times = []
    
    print(f"\n{'='*60}")
    print(f"Training with {'Model Parallel' if use_model_parallel else 'Single Device'}")
    if use_model_parallel:
        print(f"Device0: {model.device0}, Device1: {model.device1}")
    else:
        print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            if use_model_parallel:
                images = images.to(model.device0)  # 输入在device0
                labels = labels.to(model.device1)  # 标签在device1（最终输出设备）
            else:
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
        
        scheduler.step()
        
        # 测试
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if use_model_parallel:
                    images = images.to(model.device0)
                    labels = labels.to(model.device1)
                else:
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
    
    avg_time = sum(train_times) / len(train_times)
    total_time = sum(train_times)
    
    print(f"{'='*60}")
    print(f"Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Average time per epoch: {avg_time:.2f}s")
    print(f"Total training time: {total_time:.2f}s")
    print(f"{'='*60}\n")
    
    return best_test_acc, avg_time, total_time


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {device_count}")
    
    if args.device == "gpu" and torch.cuda.is_available():
        primary_device = torch.device('cuda:0')
    else:
        primary_device = torch.device('cpu')
        print("Warning: Model parallel requires GPU")
    
    train_loader, test_loader = get_data_loaders(
        args.data_root, args.batch_size, args.num_workers
    )
    
    # 创建模型
    if args.use_model_parallel and device_count >= 2:
        # 模型并行：需要至少2个GPU
        model = ModelParallelNet(num_classes=10, device_ids=list(range(min(2, device_count))))
        use_mp = True
        print(f"Using Model Parallel on GPUs {model.device_ids}")
    else:
        # 单设备模型
        if args.use_model_parallel:
            print("Warning: --use-model-parallel specified but < 2 GPUs available, using single device")
        model = SingleDeviceNet(num_classes=10)
        model = model.to(primary_device)
        use_mp = False
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    best_acc, avg_time, total_time = train_model(
        model, train_loader, test_loader, primary_device,
        args.epochs, args.lr, use_mp
    )
    
    return {
        'best_accuracy': best_acc,
        'avg_time_per_epoch': avg_time,
        'total_time': total_time,
        'use_model_parallel': use_mp
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Task2: PyTorch Model Parallel Training")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu",
                       help="Primary device: 'cpu' or 'gpu'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--use-model-parallel", action="store_true",
                       help="Use model parallel (requires >= 2 GPUs)")
    return parser


if __name__ == "__main__":
    results = main(build_argparser().parse_args())

