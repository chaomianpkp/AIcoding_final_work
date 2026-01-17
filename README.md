# 人工智能中的编程 - CIFAR-10 图像分类大作业

本项目实现了基于 CIFAR-10 数据集的图像分类系统，包含三个主要任务：PyTorch 基础实现、并行训练实践和自定义 CUDA 框架实现。

## 项目概述

CIFAR-10 数据集包含 10 个类别的 32×32 彩色图像。本项目通过三个递进的任务，从使用 PyTorch 框架到自主实现 CUDA 框架，全面展示了深度学习框架的实现原理和优化方法。

##  任务说明

### Task 1: PyTorch 基础实现

使用 PyTorch 实现卷积神经网络完成 CIFAR-10 图像分类。

**主要特性**：
- 改进的 CNN 架构（6 层卷积 + 全连接层）
- 数据增强（随机翻转、裁剪、归一化）
- 学习率调度（指数/阶梯衰减）
- 训练日志记录（JSON 格式）
- 可视化工具（自动生成训练曲线）

**快速开始**：
```bash
cd Task1
python AIHW1.py --model improved --epochs 20 --optimizer adam --device gpu --use-augmentation
```

**预期准确率**：78-82%

详细说明请参考：[Task1/README_training.md](Task1/README_training.md)

---

### Task 2: PyTorch 并行训练实践

实现数据并行和模型并行，提高 CIFAR-10 训练效率。

**主要特性**：
- 数据并行（DataParallel）：多 GPU 数据并行训练
- 模型并行（Model Parallel）：模型分布在不同 GPU 上
- 性能对比工具：自动对比不同并行方法的性能
- 详细的训练时间统计

**快速开始**：
```bash
cd Task2
# 数据并行训练
python data_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-parallel

# 模型并行训练
python model_parallel_train.py --epochs 10 --batch-size 128 --device gpu --use-model-parallel

# 性能对比
python compare_parallel.py --epochs 10 --batch-size 128
```

**预期加速比**：
- 数据并行（2 GPU）：~1.5-1.8x
- 模型并行（2 GPU）：~0.9-1.1x（适合超大模型）

详细说明请参考：[Task2/README.md](Task2/README.md)

---

### Task 3: 自定义 CUDA 框架实现

基于 CUDA、pybind11 和 Python 自主实现卷积网络框架，完成 CIFAR-10 分类。

**主要特性**：
- CUDA 实现的算子：卷积、线性、激活、池化、Softmax、交叉熵
- Python 端自动微分和优化器（SGD、Adam）
- pybind11 绑定：将 CUDA 代码转换为 Python 可调用的扩展
- 支持 CPU 和 GPU 两种模式

**编译和训练**：
```bash
cd Task3/core
python setup.py build_ext --inplace

cd ..
python train_cifar10.py --epochs 10 --batch-size 64 --optimizer adam --device gpu
```

**预期准确率**：65-75%（基础模型），可通过改进架构提升至 75-85%

详细说明请参考：[Task3/README.md](Task3/README.md)

##  环境要求

### 基础环境
- Python >= 3.8
- PyTorch >= 2.0
- torchvision >= 0.15
- numpy
- matplotlib（用于可视化）

### CUDA 环境（Task3 需要）
- CUDA Toolkit（推荐 11.0+）
- pybind11
- 支持 CUDA 的 GPU（可选，CPU 模式也可运行）

### 安装依赖

```bash
# 基础依赖
pip install torch torchvision numpy matplotlib

# Task3 额外依赖
pip install pybind11
```

##  项目结构

```
AIcoding_final_work/
├── README.md                 # 本文件
├── README(1).pdf            # 作业要求文档
│
├── Task1/                    # PyTorch 基础实现
│   ├── AIHW1.py             # 主训练脚本
│   ├── visualize_training.py  # 可视化工具
│   ├── README_training.md   # 详细使用说明
│   ├── logs/                # 训练日志（JSON）
│   └── figures/             # 可视化图片
│
├── Task2/                    # 并行训练实践
│   ├── data_parallel_train.py    # 数据并行实现
│   ├── model_parallel_train.py   # 模型并行实现
│   ├── compare_parallel.py       # 性能对比工具
│   └── README.md            # 详细使用说明
│
└── Task3/                    # 自定义 CUDA 框架
    ├── train_cifar10.py     # 训练脚本
    ├── core/                # CUDA 核心实现
    │   ├── src/             # CUDA 源文件
    │   ├── include/          # 头文件
    │   └── setup.py         # 编译配置
    ├── framework/           # Python 端框架
    └── README.md            # 详细使用说明
```

##  快速开始

### 1. Task1：PyTorch 基础实现

```bash
cd Task1
python AIHW1.py \
    --model improved \
    --epochs 20 \
    --batch-size 64 \
    --optimizer adam \
    --lr 1e-3 \
    --weight-decay 5e-4 \
    --use-augmentation \
    --use-lr-scheduler \
    --device gpu
```

### 2. Task2：并行训练

```bash
cd Task2
# 数据并行
python data_parallel_train.py --epochs 10 --device gpu --use-parallel

# 性能对比
python compare_parallel.py --epochs 10
```

### 3. Task3：自定义框架

```bash
cd Task3/core
python setup.py build_ext --inplace

cd ..
python train_cifar10.py --epochs 10 --device gpu
```


##  使用说明

每个 Task 都有详细的 README 文档：

- **Task1**: [README_training.md](Task1/README.md) - 包含训练配置、可视化说明
- **Task2**: [README.md](Task2/README.md) - 包含并行训练详细说明
- **Task3**: [README.md](Task3/README.md) - 包含编译和训练说明

##  关键文件说明

### Task1
- `AIHW1.py`: 主训练脚本，支持多种配置
- `visualize_training.py`: 训练日志可视化工具
- `logs/`: 自动保存的训练日志（JSON 格式）

### Task2
- `data_parallel_train.py`: 数据并行训练实现
- `model_parallel_train.py`: 模型并行训练实现
- `compare_parallel.py`: 自动对比不同并行方法

### Task3
- `core/`: CUDA 核心实现（需要编译）
- `framework/`: Python 端自动微分框架
- `train_cifar10.py`: 训练脚本

##  注意事项

1. **数据加载**：所有 Task 使用 `torchvision` 自动下载 CIFAR-10 数据集
2. **GPU 要求**：
   - Task1/Task2：支持 CPU 和 GPU
   - Task3：推荐 GPU，CPU 模式较慢
3. **编译**：Task3 需要先编译 CUDA 扩展（仅需一次）
4. **日志保存**：Task1 自动保存训练日志到 `logs/` 目录
5. **可视化**：Task1 自动生成可视化图片到 `figures/` 目录

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Pybind11 文档](https://pybind11.readthedocs.io/)
- [CIFAR-10 数据集](https://www.cs.toronto.edu/~kriz/cifar.html)
