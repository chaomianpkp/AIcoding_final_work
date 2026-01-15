# Task3 - 自定义CUDA框架实现 CIFAR-10

## 目录结构

- `core/`：CUDA + pybind11 实现的算子与 Tensor（复用 HW4）
- `framework/`：Python 端自动微分与优化器
- `train_cifar10.py`：CIFAR-10 训练脚本

## 编译（一次即可）

进入 `core/` 编译扩展：

```bash
cd /Users/chaomian/Desktop/人工智能中的编程/最终作业/Task3/core
python setup.py build_ext --inplace
```

## 训练

```bash
cd /Users/chaomian/Desktop/人工智能中的编程/最终作业/Task3
python train_cifar10.py --epochs 5 --batch-size 64 --optimizer adam --device cpu
```

> 如果机器有 GPU 并已配置 CUDA，可将 `--device` 改为 `gpu`。

## 说明

- 仅卷积、线性、激活、池化、softmax 与交叉熵由 CUDA 实现。
- Python 端负责计算图、反向传播与优化器更新。
- 数据加载使用 `torchvision`，不参与模型训练的核心框架实现。

