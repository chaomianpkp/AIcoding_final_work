# Task2: PyTorch 并行训练实践

实现数据并行和模型并行，提高 CIFAR-10 训练效率。

## 快速开始

### 数据并行训练

```bash
# 单GPU训练（baseline）
python data_parallel_train.py --epochs 10 --batch-size 128 --device gpu

# 数据并行训练（自动检测并使用多GPU）
python data_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-parallel
```

### 模型并行训练

```bash
# 模型并行训练（需要至少2个GPU）
python model_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-model-parallel
```

### 性能对比

```bash
# 自动对比单GPU、数据并行、模型并行
python compare_parallel.py --epochs 10 --batch-size 512
```

## 线性缩放法则

对比脚本自动应用线性缩放法则（Linear Scaling Rule）：

- **Batch Size**: `base_batch_size × num_gpus`
- **Learning Rate**: `base_lr × num_gpus × scaling_factor`

例如：单GPU batch_size=64, lr=1e-3 → 2GPU batch_size=128, lr=2e-3

```bash
# 使用保守的学习率缩放（0.9）
python compare_parallel.py --epochs 10 --batch-size 64 --lr 1e-3 --lr-scaling-factor 0.9
```

## 命令行参数

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 128 | 批次大小（单GPU） |
| `--lr` | 1e-3 | 学习率（单GPU） |
| `--device` | `gpu` | 设备：`cpu` 或 `gpu` |
| `--lr-scaling-factor` | 1.0 | 学习率缩放因子 |

### 数据并行

| 参数 | 说明 |
|------|------|
| `--use-parallel` | 启用数据并行（需要多GPU） |

### 模型并行

| 参数 | 说明 |
|------|------|
| `--use-model-parallel` | 启用模型并行（需要≥2个GPU） |


## 文件说明

- `data_parallel_train.py`: 数据并行训练脚本
- `model_parallel_train.py`: 模型并行训练脚本
- `compare_parallel.py`: 性能对比脚本（自动应用线性缩放法则）


## 注意事项

1. **GPU要求**：数据并行建议2个或更多GPU，模型并行需要至少2个GPU
2. **Batch Size**：数据并行时，有效批次大小 = `batch_size × num_gpus`
3. **线性缩放**：对比脚本会自动应用线性缩放法则，保持训练动态一致
