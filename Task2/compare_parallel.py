# Task2: 对比数据并行、模型并行和单GPU的性能
# 用于生成性能对比报告

import argparse
import time
import torch
from data_parallel_train import main as train_data_parallel
from model_parallel_train import main as train_model_parallel


def compare_methods(args):
    """对比不同并行方法的性能"""
    results = {}
    
    print("\n" + "="*70)
    print("Task2: PyTorch Parallel Training Performance Comparison")
    print("="*70 + "\n")
    
    # 1. 单GPU训练（无并行）
    print("="*70)
    print("Method 1: Single GPU (Baseline)")
    print("="*70)
    single_gpu_args = argparse.Namespace(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="gpu",
        seed=args.seed,
        num_workers=args.num_workers,
        use_parallel=False
    )
    results['single_gpu'] = train_data_parallel(single_gpu_args)
    
    # 2. 数据并行训练
    if torch.cuda.device_count() > 1:
        print("\n" + "="*70)
        print("Method 2: Data Parallel (Multi-GPU)")
        print("="*70)
        data_parallel_args = argparse.Namespace(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device="gpu",
            seed=args.seed,
            num_workers=args.num_workers,
            use_parallel=True
        )
        results['data_parallel'] = train_data_parallel(data_parallel_args)
    else:
        print("\nWarning: Only 1 GPU available, skipping Data Parallel test")
        results['data_parallel'] = None
    
    # 3. 模型并行训练
    if torch.cuda.device_count() >= 2:
        print("\n" + "="*70)
        print("Method 3: Model Parallel (Multi-GPU)")
        print("="*70)
        model_parallel_args = argparse.Namespace(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device="gpu",
            seed=args.seed,
            num_workers=args.num_workers,
            use_model_parallel=True
        )
        results['model_parallel'] = train_model_parallel(model_parallel_args)
    else:
        print("\nWarning: < 2 GPUs available, skipping Model Parallel test")
        results['model_parallel'] = None
    
    # 打印对比结果
    print("\n" + "="*70)
    print("Performance Comparison Summary")
    print("="*70)
    print(f"{'Method':<20} {'Accuracy':<12} {'Time/Epoch':<15} {'Total Time':<15} {'Speedup':<10}")
    print("-"*70)
    
    baseline_time = results['single_gpu']['avg_time_per_epoch']
    
    # 单GPU
    print(f"{'Single GPU':<20} "
          f"{results['single_gpu']['best_accuracy']:>6.2f}%     "
          f"{baseline_time:>8.2f}s       "
          f"{results['single_gpu']['total_time']:>10.2f}s   "
          f"{'1.00x':<10}")
    
    # 数据并行
    if results['data_parallel']:
        dp_speedup = baseline_time / results['data_parallel']['avg_time_per_epoch']
        print(f"{'Data Parallel':<20} "
              f"{results['data_parallel']['best_accuracy']:>6.2f}%     "
              f"{results['data_parallel']['avg_time_per_epoch']:>8.2f}s       "
              f"{results['data_parallel']['total_time']:>10.2f}s   "
              f"{dp_speedup:>6.2f}x")
    
    # 模型并行
    if results['model_parallel']:
        mp_speedup = baseline_time / results['model_parallel']['avg_time_per_epoch']
        print(f"{'Model Parallel':<20} "
              f"{results['model_parallel']['best_accuracy']:>6.2f}%     "
              f"{results['model_parallel']['avg_time_per_epoch']:>8.2f}s       "
              f"{results['model_parallel']['total_time']:>10.2f}s   "
              f"{mp_speedup:>6.2f}x")
    
    print("="*70)
    print("\nNote: Speedup is calculated relative to single GPU baseline")
    
    # 分析结论
    print("\n" + "="*70)
    print("Analysis:")
    print("="*70)
    print("1. Data Parallel: 适合模型较小、显存充足的情况")
    print("   - 优点：实现简单，对于大多数模型有较好的加速效果")
    print("   - 缺点：需要将完整模型复制到每个GPU，显存消耗大")
    print()
    print("2. Model Parallel: 适合模型太大无法放入单GPU的情况")
    print("   - 优点：可以训练超大模型，显存利用率高")
    print("   - 缺点：由于数据传输开销，加速效果可能不明显")
    print()
    print("3. 对于CIFAR-10这类小数据集和中等规模模型，Data Parallel通常更优")
    print("="*70 + "\n")


def build_argparser():
    parser = argparse.ArgumentParser(description="Task2: Compare different parallel methods")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (for comparison)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    return parser


if __name__ == "__main__":
    compare_methods(build_argparser().parse_args())

