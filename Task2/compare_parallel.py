# Task2: 对比数据并行、模型并行和单GPU的性能
# 用于生成性能对比报告

import argparse
import time
import torch
from data_parallel_train import main as train_data_parallel
from model_parallel_train import main as train_model_parallel


def apply_linear_scaling_rule(base_batch_size, base_lr, num_gpus, lr_scaling_factor=1.0):
    """
    应用线性缩放法则 (Linear Scaling Rule)
    
    原理：
    - Batch size 线性缩放: new_batch_size = base_batch_size * num_gpus
    - Learning rate 线性缩放: new_lr = base_lr * num_gpus * lr_scaling_factor
    
    Args:
        base_batch_size: 单GPU的batch size
        base_lr: 单GPU的学习率
        num_gpus: GPU数量
        lr_scaling_factor: 学习率缩放因子（默认1.0，可以设为0.9或0.95更保守）
    
    Returns:
        scaled_batch_size, scaled_lr
    """
    scaled_batch_size = base_batch_size * num_gpus
    scaled_lr = base_lr * num_gpus * lr_scaling_factor
    return scaled_batch_size, scaled_lr


def compare_methods(args):
    """对比不同并行方法的性能（应用线性缩放法则）"""
    results = {}
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print("\n" + "="*70)
    print("Task2: PyTorch Parallel Training Performance Comparison")
    print("Linear Scaling Rule Applied")
    print("="*70 + "\n")
    
    # 打印缩放规则说明
    print("Linear Scaling Rule:")
    print(f"  Single GPU: batch_size={args.batch_size}, lr={args.lr:.0e}")
    if device_count > 1:
        scaled_bs, scaled_lr = apply_linear_scaling_rule(
            args.batch_size, args.lr, device_count, args.lr_scaling_factor
        )
        print(f"  {device_count} GPUs: batch_size={scaled_bs} (={args.batch_size}×{device_count}), "
              f"lr={scaled_lr:.0e} (={args.lr:.0e}×{device_count}×{args.lr_scaling_factor:.2f})")
    print()
    
    # 1. 单GPU训练（无并行）- 基线
    print("="*70)
    print("Method 1: Single GPU (Baseline)")
    print("="*70)
    print(f"Configuration: batch_size={args.batch_size}, lr={args.lr:.0e}")
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
    
    # 2. 数据并行训练（应用线性缩放法则）
    if device_count > 1:
        print("\n" + "="*70)
        print(f"Method 2: Data Parallel ({device_count} GPUs)")
        print("="*70)
        
        # 应用线性缩放法则
        scaled_batch_size, scaled_lr = apply_linear_scaling_rule(
            args.batch_size, args.lr, device_count, args.lr_scaling_factor
        )
        
        print(f"Configuration (Linear Scaling):")
        print(f"  batch_size: {args.batch_size} → {scaled_batch_size} "
              f"(×{device_count}, each GPU: {scaled_batch_size // device_count})")
        print(f"  learning_rate: {args.lr:.0e} → {scaled_lr:.0e} "
              f"(×{device_count}×{args.lr_scaling_factor:.2f})")
        print()
        
        data_parallel_args = argparse.Namespace(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=scaled_batch_size,
            lr=scaled_lr,
            device="gpu",
            seed=args.seed,
            num_workers=args.num_workers,
            use_parallel=True
        )
        results['data_parallel'] = train_data_parallel(data_parallel_args)
        results['data_parallel']['scaled_batch_size'] = scaled_batch_size
        results['data_parallel']['scaled_lr'] = scaled_lr
    else:
        print("\nWarning: Only 1 GPU available, skipping Data Parallel test")
        results['data_parallel'] = None
    
    # 3. 模型并行训练（应用线性缩放法则）
    if device_count >= 2:
        print("\n" + "="*70)
        print(f"Method 3: Model Parallel ({device_count} GPUs)")
        print("="*70)
        
        # 应用线性缩放法则
        scaled_batch_size, scaled_lr = apply_linear_scaling_rule(
            args.batch_size, args.lr, device_count, args.lr_scaling_factor
        )
        
        print(f"Configuration (Linear Scaling):")
        print(f"  batch_size: {args.batch_size} → {scaled_batch_size} (×{device_count})")
        print(f"  learning_rate: {args.lr:.0e} → {scaled_lr:.0e} "
              f"(×{device_count}×{args.lr_scaling_factor:.2f})")
        print()
        
        model_parallel_args = argparse.Namespace(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=scaled_batch_size,
            lr=scaled_lr,
            device="gpu",
            seed=args.seed,
            num_workers=args.num_workers,
            use_model_parallel=True
        )
        results['model_parallel'] = train_model_parallel(model_parallel_args)
        results['model_parallel']['scaled_batch_size'] = scaled_batch_size
        results['model_parallel']['scaled_lr'] = scaled_lr
    else:
        print("\nWarning: < 2 GPUs available, skipping Model Parallel test")
        results['model_parallel'] = None
    
    # 打印对比结果
    print("\n" + "="*80)
    print("Performance Comparison Summary")
    print("="*80)
    print(f"{'Method':<20} {'Batch Size':<12} {'LR':<12} {'Accuracy':<12} "
          f"{'Time/Epoch':<12} {'Speedup':<10}")
    print("-"*80)
    
    baseline_time = results['single_gpu']['avg_time_per_epoch']
    
    # 单GPU
    print(f"{'Single GPU':<20} "
          f"{args.batch_size:<12} "
          f"{args.lr:.0e}{'':>4} "
          f"{results['single_gpu']['best_accuracy']:>6.2f}%     "
          f"{baseline_time:>8.2f}s   "
          f"{'1.00x':<10}")
    
    # 数据并行
    if results['data_parallel']:
        dp_speedup = baseline_time / results['data_parallel']['avg_time_per_epoch']
        scaled_bs = results['data_parallel'].get('scaled_batch_size', 'N/A')
        scaled_lr = results['data_parallel'].get('scaled_lr', 'N/A')
        print(f"{'Data Parallel':<20} "
              f"{scaled_bs:<12} "
              f"{scaled_lr:.0e}{'':>4} "
              f"{results['data_parallel']['best_accuracy']:>6.2f}%     "
              f"{results['data_parallel']['avg_time_per_epoch']:>8.2f}s   "
              f"{dp_speedup:>6.2f}x")
    
    # 模型并行
    if results['model_parallel']:
        mp_speedup = baseline_time / results['model_parallel']['avg_time_per_epoch']
        scaled_bs = results['model_parallel'].get('scaled_batch_size', 'N/A')
        scaled_lr = results['model_parallel'].get('scaled_lr', 'N/A')
        print(f"{'Model Parallel':<20} "
              f"{scaled_bs:<12} "
              f"{scaled_lr:.0e}{'':>4} "
              f"{results['model_parallel']['best_accuracy']:>6.2f}%     "
              f"{results['model_parallel']['avg_time_per_epoch']:>8.2f}s   "
              f"{mp_speedup:>6.2f}x")
    
    print("="*80)
    print("\nLinear Scaling Rule Applied:")
    print(f"  - Batch size scaled by {device_count}x (if multi-GPU)")
    print(f"  - Learning rate scaled by {device_count}x × {args.lr_scaling_factor:.2f}")
    print("  - Speedup calculated relative to single GPU baseline")
    print("\nNote: Linear scaling helps maintain training dynamics similar to single GPU")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Task2: Compare different parallel methods (with Linear Scaling Rule)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Linear Scaling Rule:
  When using N GPUs, both batch size and learning rate are scaled:
  - batch_size = base_batch_size × N
  - learning_rate = base_lr × N × scaling_factor (default: 1.0)
  
  This maintains similar training dynamics to single GPU training.

Example:
  python compare_parallel.py --epochs 5 --batch-size 64 --lr 1e-3
  # With 2 GPUs: batch_size=128, lr=2e-3 (with scaling_factor=1.0)
        """
    )
    parser.add_argument("--data-root", type=str, default="./data", 
                       help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs (for comparison)")
    parser.add_argument("--batch-size", type=int, default=128, 
                       help="Base batch size (single GPU). Will be scaled for multi-GPU")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Base learning rate (single GPU). Will be scaled for multi-GPU")
    parser.add_argument("--lr-scaling-factor", type=float, default=1.0,
                       help="Learning rate scaling factor (default: 1.0, can use 0.9 or 0.95 for more conservative)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, 
                       help="Number of data loading workers")
    return parser


if __name__ == "__main__":
    compare_methods(build_argparser().parse_args())

