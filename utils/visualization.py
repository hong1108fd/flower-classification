"""
可视化工具
绘制训练曲线和结果对比
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


def plot_training_curves(
    history: Dict[str, List],
    save_path: str,
    title: str = "Training History"
):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_hyperparameter_comparison(
    results: List[Dict],
    save_path: str
):
    """绘制超参数对比图"""
    # 提取数据
    lrs = [r['lr'] for r in results]
    bss = [r['batch_size'] for r in results]
    accs = [r['best_val_acc'] for r in results]

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按学习率分组
    lr_groups = {}
    for r in results:
        lr = r['lr']
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(r['best_val_acc'])

    x = range(len(lr_groups))
    means = [np.mean(v) for v in lr_groups.values()]
    stds = [np.std(v) for v in lr_groups.values()]
    labels = [f'lr={lr}' for lr in lr_groups.keys()]

    axes[0].bar(x, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[0].set_title('Effect of Learning Rate', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 按batch size分组
    bs_groups = {}
    for r in results:
        bs = r['batch_size']
        if bs not in bs_groups:
            bs_groups[bs] = []
        bs_groups[bs].append(r['best_val_acc'])

    x = range(len(bs_groups))
    means = [np.mean(v) for v in bs_groups.values()]
    stds = [np.std(v) for v in bs_groups.values()]
    labels = [f'bs={bs}' for bs in bs_groups.keys()]

    axes[1].bar(x, means, yerr=stds, capsize=5, color=['#d62728', '#9467bd', '#8c564b'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Effect of Batch Size', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Hyperparameter comparison saved to {save_path}")


def plot_ablation_comparison(
    results: Dict,
    save_path: str
):
    """绘制预训练消融对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['Pretrained', 'From Scratch']
    val_accs = [
        results['pretrained']['best_val_accuracy'],
        results['scratch']['best_val_accuracy']
    ]
    test_accs = [
        results['pretrained']['test_accuracy'],
        results['scratch']['test_accuracy']
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Accuracy', color='#1f77b4')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Pretrained vs From Scratch Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Ablation comparison saved to {save_path}")


def plot_attention_comparison(
    results: Dict,
    save_path: str
):
    """绘制注意力机制对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    val_accs = [results[m]['best_val_accuracy'] for m in models]
    test_accs = [results[m]['test_accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_accs, width, label='Val Accuracy', color='#1f77b4')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Attention Mechanism Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, rotation=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention comparison saved to {save_path}")


def generate_summary_table(results_dir: str, output_path: str):
    """生成结果汇总表格"""
    results_dir = Path(results_dir)

    # 收集所有结果
    summary = []

    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / 'results.json'
            config_file = exp_dir / 'config.json'

            if results_file.exists() and config_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                with open(config_file, 'r') as f:
                    config = json.load(f)

                summary.append({
                    'Experiment': exp_dir.name,
                    'Model': config.get('model', 'N/A'),
                    'Pretrained': config.get('pretrained', 'N/A'),
                    'LR': config.get('lr', 'N/A'),
                    'Batch Size': config.get('batch_size', 'N/A'),
                    'Epochs': config.get('epochs', 'N/A'),
                    'Best Val Acc': f"{results.get('best_val_accuracy', 0):.2f}%",
                    'Test Acc': f"{results.get('test_accuracy', 0):.2f}%"
                })

    # 打印表格
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    if summary:
        # 表头
        headers = summary[0].keys()
        print("  ".join(f"{h:15}" for h in headers))
        print("-" * 100)

        for row in summary:
            print("  ".join(f"{str(v):15}" for v in row.values()))

    # 保存为JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualization Tools')
    parser.add_argument('--results_dir', type=str, default='./outputs', help='结果目录')
    parser.add_argument('--output_dir', type=str, default='./figures', help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 生成汇总
    generate_summary_table(args.results_dir, os.path.join(args.output_dir, 'summary.json'))
