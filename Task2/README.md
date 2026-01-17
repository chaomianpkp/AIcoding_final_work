# Task2: PyTorch 并行训练实践

## 任务目标

学习并实现 PyTorch 的数据并行（Data Parallel）和模型并行（Model Parallel）机制，以提高 CIFAR-10 训练效率，并对比不同并行方法的性能。

## 目录结构

```
Task2/
├── data_parallel_train.py    # 数据并行实现
├── model_parallel_train.py   # 模型并行实现
├── compare_parallel.py       # 性能对比脚本
└── README.md                 # 本文件
```

## 实现方法

### 1. 数据并行（Data Parallel）

**原理**：将批次数据分割到多个GPU上，每个GPU处理一部分数据，然后在主GPU上汇总梯度并更新参数。

**优点**：
- 实现简单（使用 `nn.DataParallel`）
- 对于大多数模型有较好的加速效果
- 适合模型较小、显存充足的情况

**缺点**：
- 需要将完整模型复制到每个GPU
- 显存消耗大
- 主GPU需要汇总梯度，可能成为瓶颈

**使用方法**：
```bash
# 单GPU训练（baseline）
python data_parallel_train.py --epochs 10 --batch-size 128 --device gpu

# 数据并行训练（自动检测并使用多GPU）
python data_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-parallel
```

### 2. 模型并行（Model Parallel）

**原理**：将模型的不同部分分布在不同GPU上，数据需要在GPU间传递。

**优点**：
- 可以训练超大模型（单GPU放不下）
- 显存利用率高
- 每个GPU只需要存储模型的一部分

**缺点**：
- 由于GPU间数据传输开销，加速效果可能不明显
- 实现相对复杂
- 需要手动管理数据和模型在不同GPU上的分布

**使用方法**：
```bash
# 模型并行训练（需要至少2个GPU）
python model_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-model-parallel
```

## 性能对比

运行对比脚本，自动测试所有方法并生成报告：

```bash
python compare_parallel.py --epochs 10 --batch-size 128
```

对比脚本会依次运行：
1. 单GPU训练（baseline）
2. 数据并行训练（如果有多GPU）
3. 模型并行训练（如果有≥2个GPU）

并输出性能对比表，包括：
- 准确率
- 每个epoch的训练时间
- 总训练时间
- 相对加速比

## 命令行参数

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-root` | str | `./data` | 数据集根目录 |
| `--epochs` | int | 10 | 训练轮数 |
| `--batch-size` | int | 128 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--device` | str | `gpu` | 设备：`cpu` 或 `gpu` |
| `--seed` | int | 42 | 随机种子 |
| `--num-workers` | int | 2 | 数据加载的worker数量 |

### 数据并行特有参数

| 参数 | 说明 |
|------|------|
| `--use-parallel` | 启用数据并行（需要多GPU） |

### 模型并行特有参数

| 参数 | 说明 |
|------|------|
| `--use-model-parallel` | 启用模型并行（需要≥2个GPU） |

## 使用示例

### 示例1：单GPU训练（用于对比baseline）

```bash
python data_parallel_train.py --epochs 20 --batch-size 128 --device gpu
```

### 示例2：数据并行训练

```bash
python data_parallel_train.py --epochs 20 --batch-size 128 --device gpu --use-parallel
```

### 示例3：模型并行训练

```bash
python model_parallel_train.py --epochs 20 --batch-size 128 --device gpu --use-model-parallel
```

### 示例4：完整对比实验

```bash
# 快速对比（5个epoch）
python compare_parallel.py --epochs 5 --batch-size 128

# 完整对比（20个epoch，需要较长时间）
python compare_parallel.py --epochs 20 --batch-size 128
```

## 预期结果

### 性能对比（基于2个GPU）

| 方法 | 准确率 | 时间/Epoch | 相对加速 |
|------|--------|-----------|----------|
| 单GPU | ~78-82% | 100s | 1.00x |
| 数据并行 | ~78-82% | 55-65s | ~1.5-1.8x |
| 模型并行 | ~78-82% | 90-110s | ~0.9-1.1x |

**注意**：
- 实际性能取决于硬件配置（GPU型号、显存、NVLink等）
- 数据并行通常优于模型并行（对于CIFAR-10这类任务）
- 模型并行更适合超大模型场景

## 实现细节

### 数据并行实现

```python
# 使用PyTorch的DataParallel包装模型
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

DataParallel会自动：
- 将输入batch分割到多个GPU
- 在每个GPU上执行前向传播
- 在主GPU（GPU 0）上汇总梯度
- 更新参数并同步到所有GPU

### 模型并行实现

```python
# 手动将模型的不同部分放在不同GPU上
self.features_part1 = nn.Sequential(...).to(device0)
self.features_part2 = nn.Sequential(...).to(device1)
self.classifier = nn.Sequential(...).to(device1)

# 在前向传播中手动传输数据
x = x.to(device0)
x = self.features_part1(x)
x = x.to(device1)  # GPU间传输
x = self.features_part2(x)
x = self.classifier(x)
```

## 注意事项

1. **GPU要求**：
   - 数据并行：建议2个或更多GPU
   - 模型并行：需要至少2个GPU
   - 如果只有1个GPU，可以运行但无法测试并行效果

2. **显存限制**：
   - 数据并行：每个GPU需要存储完整模型
   - 模型并行：每个GPU只存储模型的一部分

3. **数据传输开销**：
   - 模型并行在GPU间传输数据，可能影响性能
   - 使用NVLink的高速互联可以减少开销

4. **批次大小**：
   - 数据并行时，有效批次大小 = `batch_size * num_gpus`
   - 可能需要相应调整学习率

## 实验报告建议

在报告中应包括：
1. **实验环境**：GPU型号、数量、CUDA版本等
2. **实验结果**：准确率、训练时间对比表
3. **性能分析**：分析为什么数据并行通常优于模型并行
4. **可视化**：训练曲线对比图、速度提升图
5. **结论**：总结不同并行方法的适用场景

