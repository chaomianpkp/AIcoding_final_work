# 可视化训练日志
# 用于绘制训练过程中的loss、准确率等曲线

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_log(log_path):
    """加载训练日志"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_training_curves(log_data, save_path=None, show=False):
    """绘制训练曲线"""
    history = log_data['training_history']
    config = log_data['config']
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    learning_rate = [h['learning_rate'] for h in history]
    
    # 创建1x2的子图
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(f'Training Curves - {config["model"]} Model ({config["optimizer"]})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Loss曲线
    axes[0].plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # 2. 准确率曲线
    axes[1].plot(epochs, train_acc, 'g-', linewidth=2, label='Train Accuracy', alpha=0.8)
    axes[1].plot(epochs, test_acc, 'r-', linewidth=2, label='Test Accuracy', alpha=0.8)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim([0, 100])
    
    
    plt.tight_layout()
    
    # 自动保存（如果未指定路径，则自动生成）
    if save_path is None:
        # 基于配置自动生成文件名
        model_name = config['model']
        optimizer_name = config['optimizer']
        aug_suffix = "_aug" if config.get('use_augmentation', False) else ""
        save_path = f"./figures/training_curves_{model_name}_{optimizer_name}{aug_suffix}.png"
    
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_experiment_label(config, index=None):
    """生成实验标签"""
    parts = []
    
    # 模型类型
    if config.get('model') == 'improved':
        parts.append('Improved')
    elif config.get('model') == 'simple':
        parts.append('Simple')
    
    # 优化器
    opt_map = {'adam': 'Adam', 'sgd': 'SGD'}
    parts.append(opt_map.get(config.get('optimizer', 'adam'), 'Unknown'))
    
    # 数据增强
    if config.get('use_augmentation', False):
        parts.append('+Aug')
    
    # 学习率调度
    if config.get('use_lr_scheduler', False):
        parts.append('+LR_Sched')
    
    # 批次大小
    if config.get('batch_size'):
        parts.append(f'BS{config["batch_size"]}')
    
    # 学习率
    if config.get('lr'):
        lr_str = f"{config['lr']:.0e}".replace('e-0', 'e-')
        parts.append(f'LR{lr_str}')
    
    label = ' '.join(parts)
    
    # 如果有多个实验，添加索引
    if index is not None:
        label = f"Exp{index+1}: {label}"
    
    return label


def compare_logs(log_paths, save_path=None, show=False, compare_loss=True):
    """对比多个训练日志 - 增强版"""
    # 加载所有日志数据
    all_logs = []
    for log_path in log_paths:
        try:
            log_data = load_log(log_path)
            log_data['file_path'] = log_path
            all_logs.append(log_data)
        except Exception as e:
            print(f"Warning: Failed to load {log_path}: {e}")
            continue
    
    if len(all_logs) == 0:
        print("Error: No valid log files found")
        return
    
    # 设置颜色和样式
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_logs)))
    base_styles = ['-', '--', '-.', ':']
    linestyles = [base_styles[i % len(base_styles)] for i in range(len(all_logs))]
    
    # 创建子图：准确率对比 + (可选) Loss对比
    n_cols = 3 if compare_loss else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
    if n_cols == 2:
        axes = [axes[0], axes[1]]
    
    plot_idx = 0
    
    # 1. Loss对比（如果启用）
    if compare_loss:
        for i, log_data in enumerate(all_logs):
            history = log_data['training_history']
            config = log_data['config']
            
            epochs = [h['epoch'] for h in history]
            train_loss = [h['train_loss'] for h in history]
            
            label = generate_experiment_label(config, i)
            axes[plot_idx].plot(epochs, train_loss, color=colors[i], 
                              linewidth=2.5, label=label, alpha=0.8)
        
        axes[plot_idx].set_xlabel('Epoch', fontsize=12)
        axes[plot_idx].set_ylabel('Training Loss', fontsize=12)
        axes[plot_idx].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
        axes[plot_idx].legend(fontsize=9, loc='upper right')
        plot_idx += 1
    
    # 2. 训练准确率对比
    for i, log_data in enumerate(all_logs):
        history = log_data['training_history']
        config = log_data['config']
        
        epochs = [h['epoch'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        
        label = generate_experiment_label(config, i)
        axes[plot_idx].plot(epochs, train_acc, color=colors[i], 
                          linewidth=2.5, linestyle='--', label=f'{label} (Train)', alpha=0.8)
    
    axes[plot_idx].set_xlabel('Epoch', fontsize=12)
    axes[plot_idx].set_ylabel('Train Accuracy (%)', fontsize=12)
    axes[plot_idx].set_title('Train Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
    axes[plot_idx].legend(fontsize=9, loc='lower right')
    axes[plot_idx].set_ylim([0, 100])
    plot_idx += 1
    
    # 3. 测试准确率对比
    for i, log_data in enumerate(all_logs):
        history = log_data['training_history']
        config = log_data['config']
        
        epochs = [h['epoch'] for h in history]
        test_acc = [h['test_acc'] for h in history]
        
        label = generate_experiment_label(config, i)
        timing = log_data.get('timing', {})
        best_acc = timing.get('best_test_acc', test_acc[-1] if test_acc else 0)
        
        # 在标签中添加最佳准确率
        full_label = f'{label} (Best: {best_acc:.2f}%)'
        axes[plot_idx].plot(epochs, test_acc, color=colors[i], 
                          linewidth=2.5, label=full_label, alpha=0.8)
        
        # 标记最佳点
        best_idx = test_acc.index(max(test_acc)) if test_acc else 0
        axes[plot_idx].plot(epochs[best_idx], max(test_acc), 'o', 
                          color=colors[i], markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    axes[plot_idx].set_xlabel('Epoch', fontsize=12)
    axes[plot_idx].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[plot_idx].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[plot_idx].grid(True, alpha=0.3, linestyle='--')
    axes[plot_idx].legend(fontsize=9, loc='lower right')
    axes[plot_idx].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 自动保存（如果未指定路径，则自动生成）
    if save_path is None:
        save_path = "./figures/experiment_comparison.png"
    
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to: {save_path}")
    
    # 打印实验配置对比表
    print("\n" + "="*80)
    print("实验配置对比")
    print("="*80)
    print(f"{'实验':<15} {'模型':<10} {'优化器':<8} {'Batch':<8} {'LR':<12} {'数据增强':<10} {'最佳准确率':<12}")
    print("-"*80)
    
    for i, log_data in enumerate(all_logs):
        config = log_data['config']
        timing = log_data.get('timing', {})
        best_acc = timing.get('best_test_acc', 0)
        
        model = config.get('model', 'N/A')
        opt = config.get('optimizer', 'N/A')
        bs = config.get('batch_size', 'N/A')
        lr = config.get('lr', 'N/A')
        aug = 'Yes' if config.get('use_augmentation', False) else 'No'
        
        print(f"Exp{i+1}: {model:<10} {opt:<8} {bs:<8} {lr:<12.0e} {aug:<10} {best_acc:<12.2f}%")
    
    print("="*80 + "\n")
    
    if show:
        plt.show()
    else:
        plt.close()




def main():
    parser = argparse.ArgumentParser(
        description="Visualize training logs and compare experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 可视化单个实验
  python visualize_training.py --log-path ./logs/training_log_improved_adam_xxx.json
  
  # 对比两个或多个实验（需要至少2个文件）
  python visualize_training.py --compare ./logs/log1.json ./logs/log2.json
  
  # 对比三个实验
  python visualize_training.py --compare log1.json log2.json log3.json
  
  # 对比并排除loss曲线
  python visualize_training.py --compare log1.json log2.json --no-compare-loss
        """
    )
    parser.add_argument("--log-path", type=str, default=None,
                       help="Path to training log JSON file (single experiment)")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save the figure (default: auto-generate in ./figures/)")
    parser.add_argument("--show", action="store_true",
                       help="Display the plot (default: only save)")
    parser.add_argument("--compare", nargs='+', default=None,
                       help="Compare multiple log files (requires at least 2 files)")
    parser.add_argument("--compare-loss", action="store_true", default=True,
                       help="Include loss comparison in comparison plots (default: True)")
    parser.add_argument("--no-compare-loss", dest="compare_loss", action="store_false",
                       help="Exclude loss comparison in comparison plots")
    args = parser.parse_args()
    
    # 确定模式
    if args.compare:
        # 对比指定的日志文件（需要至少2个）
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 log files")
            print(f"Provided: {len(args.compare)} file(s)")
            return
        compare_logs(args.compare, save_path=args.save_path, show=args.show,
                    compare_loss=args.compare_loss)
    elif args.log_path:
        # 单个日志可视化
        log_data = load_log(args.log_path)
        plot_training_curves(log_data, save_path=args.save_path, show=args.show)
    else:
        parser.print_help()
        print("\nError: Please specify --log-path or --compare (with at least 2 files)")


if __name__ == "__main__":
    main()

