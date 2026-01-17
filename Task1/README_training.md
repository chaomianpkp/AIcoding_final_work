# Task1 训练说明 - 准确率提升指南

## 快速开始

### 基础配置（原始网络）
```bash
python AIHW1.py --epochs 10 --batch-size 64 --optimizer sgd --device gpu
```

### 改进配置（推荐，预期准确率75-82%）
```bash
python AIHW1.py \
    --model improved \
    --epochs 10 \
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

## 主要改进点

### 1. 改进的模型架构（ImprovedCNN）
- **更深的网络**：6个卷积层（原为2个）
- **更宽的通道**：64 → 128 → 256（原为6 → 16）
- **Dropout正则化**：防止过拟合
- **Kaiming初始化**：适用于ReLU激活函数的权重初始化

### 2. 数据增强
- **随机水平翻转**：50%概率
- **随机裁剪**：填充4像素后随机裁剪
- **数据归一化**：使用CIFAR-10的均值和标准差

### 3. 学习率调度
- **指数衰减**：每个epoch乘以衰减因子
- **阶梯衰减**：每N个epoch衰减一次

### 4. 优化器选项
- **Adam**：自适应学习率，通常收敛更快
- **SGD with Momentum**：配合momentum=0.9使用

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-root` | str | `./data` | 数据集根目录 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 64 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--weight-decay` | float | 5e-4 | L2正则化权重 |
| `--optimizer` | str | `adam` | 优化器：`sgd` 或 `adam` |
| `--device` | str | `cpu` | 设备：`cpu` 或 `gpu` |
| `--seed` | int | 42 | 随机种子 |
| `--model` | str | `improved` | 模型：`simple` 或 `improved` |
| `--use-augmentation` | flag | False | 启用数据增强 |
| `--use-lr-scheduler` | flag | False | 启用学习率调度 |
| `--lr-scheduler-mode` | str | `exp` | 调度模式：`step` 或 `exp` |
| `--lr-decay-factor` | float | 0.95 | 学习率衰减因子 |
| `--lr-step-size` | int | 5 | Step模式的步长 |

## 训练示例

### 示例1：使用改进模型（无数据增强）
```bash
python AIHW1.py --model improved --epochs 15 --optimizer adam --device gpu
```
预期准确率：~70-75%

### 示例2：添加数据增强
```bash
python AIHW1.py \
    --model improved \
    --epochs 15 \
    --optimizer adam \
    --use-augmentation \
    --device gpu
```
预期准确率：~75-80%

### 示例3：完整优化（所有改进）
```bash
python AIHW1.py \
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
预期准确率：~78-82%

### 示例4：使用SGD优化器
```bash
python AIHW1.py \
    --model improved \
    --epochs 20 \
    --optimizer sgd \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --use-augmentation \
    --use-lr-scheduler \
    --lr-scheduler-mode step \
    --lr-step-size 10 \
    --lr-decay-factor 0.1 \
    --device gpu
```

## 预期准确率对比

| 配置 | 预期准确率 | 改进点 |
|------|-----------|--------|
| 原始SimpleNet | ~55-60% | - |
| ImprovedNet（无增强） | ~70-75% | 更深的网络 + 初始化 |
| ImprovedNet + 数据增强 | ~75-80% | + 数据增强 |
| 完整优化 | ~78-82% | + 学习率调度 + 正则化 |

## 调优建议

1. **如果准确率偏低**：
   - 增加训练轮数（`--epochs 20-30`）
   - 启用数据增强（`--use-augmentation`）
   - 使用学习率调度（`--use-lr-scheduler`）

2. **如果过拟合**（训练准确率 >> 测试准确率）：
   - 增加权重衰减（`--weight-decay 1e-3`）
   - 增加Dropout率（修改代码中的`dropout=0.5`为`0.6-0.7`）

3. **如果欠拟合**（训练和测试准确率都低）：
   - 增加模型深度/宽度
   - 增加训练轮数
   - 降低学习率（`--lr 5e-4`）

4. **训练速度优化**：
   - 使用GPU（`--device gpu`）
   - 增加批次大小（`--batch-size 128`，注意内存限制）

