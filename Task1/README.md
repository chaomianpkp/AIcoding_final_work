# Task1: PyTorch CIFAR-10 图像分类

使用 PyTorch 实现卷积神经网络，完成 CIFAR-10 数据集的图像分类任务。

### 基础训练

```bash
python train_cifar10.py --epochs 10 --batch-size 64 --optimizer adam --device gpu
```

### 推荐配置（预期准确率 78-82%）

```bash
python train_cifar10.py \
    --model improved \
    --epochs 20 \
    --batch-size 64 \
    --optimizer adam \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --use-augmentation \
    --use-lr-scheduler \
    --lr-scheduler-mode exp \
    --lr-decay-factor 0.95 \
    --device gpu
```

## 模型架构

### SimpleCNN
- 2层卷积（3→6→16）
- 3层全连接（400→120→84→10）
- 参数量：~0.1M
- 预期准确率：~55-60%

### ImprovedCNN
- 6层卷积（3→64→128→256）
- 3层全连接（4096→512→256→10）
- Dropout (0.5) 正则化
- Kaiming 权重初始化
- 参数量：~1.2M
- 预期准确率：~78-82%

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `improved` | 模型：`simple` 或 `improved` |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--optimizer` | `adam` | 优化器：`sgd` 或 `adam` |
| `--lr` | 1e-3 | 学习率 |
| `--weight-decay` | 5e-4 | L2正则化权重 |
| `--device` | `cpu` | 设备：`cpu` 或 `gpu` |
| `--use-augmentation` | False | 启用数据增强 |
| `--use-lr-scheduler` | False | 启用学习率调度 |
| `--lr-scheduler-mode` | `exp` | 调度模式：`step` 或 `exp` |
| `--lr-decay-factor` | 0.95 | 学习率衰减因子 |
| `--log-dir` | `./logs` | 日志保存目录 |

## 训练日志和可视化

### 训练日志

训练过程自动记录到 JSON 文件，保存在 `./logs/` 目录。

### 可视化训练曲线

```bash
# 单个实验可视化
python visualize_training.py --log-path ./logs/training_log_improved_adam_xxx.json

# 对比多个实验
python visualize_training.py --compare \
    ./logs/training_log_improved_adam_xxx.json \
    ./logs/training_log_improved_sgd_xxx.json
```

可视化图片自动保存到 `./figures/` 目录。

## 使用示例

```bash
python train_cifar10.py \
    --model improved \
    --epochs 20 \
    --optimizer adam \
    --use-augmentation \
    --device gpu
```

## 依赖安装

```bash
pip install torch torchvision numpy matplotlib
```

## 文件说明

- `train_cifar10.py`: 主训练脚本
- `visualize_training.py`: 训练日志可视化工具
- `logs/`: 训练日志保存目录（自动创建）
- `figures/`: 可视化图片保存目录（自动创建）
