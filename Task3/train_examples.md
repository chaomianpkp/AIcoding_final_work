# 训练示例 - 提升准确率的不同配置

## 基础配置（原始，准确率~65%）

```bash
python train_cifar10.py --epochs 5 --batch-size 64 --optimizer adam --device gpu
```

## 改进配置1：使用改进的模型架构（预期+5-8%）

```bash
python train_cifar10.py \
    --model improved \
    --epochs 10 \
    --batch-size 64 \
    --optimizer adam \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --device gpu
```

**改进点**：
- 更深的网络（4个卷积层，更多通道）
- 添加隐藏层（256×8×8 → 512 → 10）
- Kaiming/Xavier初始化
- 权重衰减正则化

## 改进配置2：添加数据增强（预期+3-6%）

```bash
python train_cifar10.py \
    --model improved \
    --epochs 10 \
    --batch-size 64 \
    --optimizer adam \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --use-augmentation \
    --device gpu
```

**改进点**：
- 随机水平翻转
- 随机裁剪（填充4像素后裁剪）

## 改进配置3：添加学习率调度（预期+2-4%）

```bash
python train_cifar10.py \
    --model improved \
    --epochs 15 \
    --batch-size 64 \
    --optimizer adam \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --use-augmentation \
    --use-lr-scheduler \
    --lr-scheduler-mode exp \
    --lr-decay-factor 0.95 \
    --device gpu
```

**改进点**：
- 指数学习率衰减（每个epoch乘以0.95）

## 改进配置4：完整优化（推荐，预期75-82%）

```bash
python train_cifar10.py \
    --model improved \
    --epochs 20 \
    --batch-size 64 \
    --optimizer adam \
    --lr 5e-4 \
    --weight-decay 5e-4 \
    --use-augmentation \
    --use-lr-scheduler \
    --lr-scheduler-mode exp \
    --lr-decay-factor 0.95 \
    --device gpu
```

**所有改进点**：
- ✓ 改进的模型架构
- ✓ 数据增强
- ✓ 学习率调度
- ✓ 权重初始化优化
- ✓ 权重衰减
- ✓ 更多训练轮数

## 配置参数说明

### 模型选择
- `--model simple`: 原始SimpleCNN（2个卷积层）
- `--model improved`: 改进的ImprovedCNN（4个卷积层，更宽）

### 数据增强
- `--use-augmentation`: 启用随机翻转和裁剪

### 学习率调度
- `--use-lr-scheduler`: 启用学习率调度
- `--lr-scheduler-mode`: `exp`（指数）或 `step`（阶梯）
- `--lr-decay-factor`: 衰减因子（0.9-0.95推荐）
- `--lr-step-size`: Step模式的步长（每N个epoch衰减）

### 优化器
- `--optimizer adam`: Adam优化器（默认）
- `--optimizer sgd`: SGD优化器（可配合momentum使用）
- `--lr`: 初始学习率（推荐：3e-4 到 1e-3）
- `--weight-decay`: L2正则化（推荐：1e-4 到 5e-4）

### 训练
- `--epochs`: 训练轮数（推荐：10-20）
- `--batch-size`: 批次大小（64或128）

## 预期准确率对比

| 配置 | 预期准确率 | 训练时间 |
|------|-----------|---------|
| 基础配置 | ~65% | 快 |
| 改进配置1 | ~70-73% | 中等 |
| 改进配置2 | ~73-76% | 中等 |
| 改进配置3 | ~75-78% | 较慢 |
| 改进配置4 | ~78-82% | 慢 |

## 调优建议

1. **先从改进配置1开始**，验证模型架构改进的效果
2. **逐步添加数据增强和学习率调度**，观察准确率提升
3. **如果过拟合**（训练准确率>>测试准确率）：
   - 增加 `--weight-decay`（如5e-4）
   - 减少训练轮数
4. **如果欠拟合**（训练和测试准确率都很低）：
   - 增加训练轮数
   - 增加模型深度/宽度
   - 降低学习率

