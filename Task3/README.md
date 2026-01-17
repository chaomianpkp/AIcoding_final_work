# Task3: 自定义CUDA框架实现 CIFAR-10

使用自定义CUDA框架实现卷积神经网络，完成CIFAR-10图像分类。

## 编译

进入 `core/` 目录编译CUDA扩展（只需一次）：

```bash
cd core
python setup.py build_ext --inplace
cd ..
```

## 快速开始

### 基础训练

```bash
# CPU训练
python train_cifar10.py --epochs 10 --batch-size 64 --optimizer adam --device cpu

# GPU训练
python train_cifar10.py --epochs 10 --batch-size 64 --optimizer adam --device gpu
```

### 推荐配置

```bash
python train_cifar10.py \
    --epochs 20 \
    --batch-size 64 \
    --optimizer adam \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --device gpu
```

预期准确率：~65-70%

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--optimizer` | `adam` | 优化器：`sgd` 或 `adam` |
| `--lr` | 1e-3 | 学习率 |
| `--weight-decay` | 0.0 | L2正则化权重 |
| `--device` | `cpu` | 设备：`cpu` 或 `gpu` |
| `--seed` | 42 | 随机种子 |
| `--log-dir` | `./logs` | 日志保存目录 |

## 训练日志和可视化

训练过程自动记录到JSON文件，保存在 `./logs/` 目录。

使用Task1的可视化脚本查看训练曲线：

```bash
# 单个实验可视化
python ../Task1/visualize_training.py --log-path ./logs/training_log_xxx.json

# 对比多个实验
python ../Task1/visualize_training.py --compare \
    ./logs/training_log_adam_xxx.json \
    ./logs/training_log_sgd_xxx.json
```

## 实现说明

- **CUDA算子**：卷积、线性、激活、池化、softmax、交叉熵在CUDA中实现
- **Python端**：自动微分、反向传播、优化器更新在Python中实现
- **数据加载**：使用 `torchvision` 加载CIFAR-10数据集

## 目录结构

- `core/`: CUDA + pybind11 实现的算子
- `framework/`: Python端自动微分与优化器
- `train_cifar10.py`: 训练脚本
