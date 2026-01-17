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
    """绘制训练曲线（不包含时间相关图表）"""
    history = log_data['training_history']
    config = log_data['config']
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    test_acc = [h['test_acc'] for h in history]
    learning_rate = [h['learning_rate'] for h in history]
    
    # 创建1x2的子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
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


def compare_logs(log_paths, save_path=None, show=False):
    """对比多个训练日志"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for log_path in log_paths:
        log_data = load_log(log_path)
        history = log_data['training_history']
        config = log_data['config']
        
        epochs = [h['epoch'] for h in history]
        train_acc = [h['train_acc'] for h in history]
        test_acc = [h['test_acc'] for h in history]
        
        label = f"{config['model']}_{config['optimizer']}"
        if config.get('use_augmentation', False):
            label += "_aug"
        
        axes[0].plot(epochs, train_acc, '--', linewidth=2, label=f'{label} (Train)', alpha=0.7)
        axes[1].plot(epochs, test_acc, '-', linewidth=2, label=f'{label} (Test)', alpha=0.7)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Train Accuracy (%)', fontsize=12)
    axes[0].set_title('Train Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim([0, 100])
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Test Accuracy (%)', fontsize=12)
    axes[1].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # 自动保存（如果未指定路径，则自动生成）
    if save_path is None:
        save_path = "./figures/comparison.png"
    
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Created directory: {save_dir}")
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison figure saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training logs")
    parser.add_argument("--log-path", type=str, required=True,
                       help="Path to training log JSON file")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save the figure (default: auto-generate in ./figures/)")
    parser.add_argument("--show", action="store_true",
                       help="Display the plot (default: only save)")
    parser.add_argument("--compare", nargs='+', default=None,
                       help="Compare multiple log files (provide multiple log paths)")
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        compare_logs(args.compare, save_path=args.save_path, show=args.show)
    else:
        # 单个日志可视化
        log_data = load_log(args.log_path)
        plot_training_curves(log_data, save_path=args.save_path, show=args.show)


if __name__ == "__main__":
    main()

