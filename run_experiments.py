"""
实验配置脚本
一键运行所有实验
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

# 实验配置
EXPERIMENTS = {
    # Baseline实验
    'baseline_resnet18': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01
    },

    # 超参数分析
    'hp_lr_001': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.001
    },
    'hp_lr_01': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01
    },
    'hp_lr_1': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.1
    },
    'hp_bs_16': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 16,
        'lr': 0.01
    },
    'hp_bs_64': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 64,
        'lr': 0.01
    },
    'hp_epochs_30': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 30,
        'batch_size': 32,
        'lr': 0.01
    },
    'hp_epochs_100': {
        'model': 'resnet18',
        'pretrained': True,
        'epochs': 100,
        'batch_size': 32,
        'lr': 0.01
    },

    # 预训练消融实验
    'ablation_scratch': {
        'model': 'resnet18',
        'pretrained': False,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01
    },

    # 注意力机制实验
    'attention_se_resnet18': {
        'model': 'se_resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01
    },
    'attention_cbam_resnet18': {
        'model': 'cbam_resnet18',
        'pretrained': True,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01
    }
}


def run_single_experiment(exp_name: str, config: dict, base_args: list):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"{'='*60}")

    # 构建命令
    cmd = ['python', 'train.py'] + base_args
    cmd.extend(['--exp_name', exp_name])

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"Command: {' '.join(cmd)}")

    # 运行
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_all_experiments(output_dir: str = './outputs'):
    """运行所有实验"""
    print("Starting all experiments...")
    print(f"Total experiments: {len(EXPERIMENTS)}")

    # 基础参数
    base_args = [
        '--data_dir', './data',
        '--output_dir', output_dir,
        '--scheduler', 'cosine',
        '--num_workers', '4'
    ]

    results = {}

    for exp_name, config in EXPERIMENTS.items():
        success = run_single_experiment(exp_name, config, base_args)
        results[exp_name] = 'success' if success else 'failed'
        print(f"\nExperiment {exp_name}: {results[exp_name]}")

    # 保存结果
    with open(os.path.join(output_dir, 'experiment_log.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)

    return results


def run_quick_experiments(output_dir: str = './outputs'):
    """运行快速实验（减少epochs）用于测试"""
    print("Running quick experiments for testing...")

    # 修改配置，减少epochs
    quick_experiments = {
        'baseline_resnet18': {
            'model': 'resnet18',
            'pretrained': True,
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.01
        },
        'ablation_scratch': {
            'model': 'resnet18',
            'pretrained': False,
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.01
        },
        'attention_se_resnet18': {
            'model': 'se_resnet18',
            'pretrained': True,
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.01
        },
        'attention_cbam_resnet18': {
            'model': 'cbam_resnet18',
            'pretrained': True,
            'epochs': 10,
            'batch_size': 32,
            'lr': 0.01
        }
    }

    base_args = [
        '--data_dir', './data',
        '--output_dir', output_dir,
        '--scheduler', 'cosine',
        '--num_workers', '4'
    ]

    results = {}

    for exp_name, config in quick_experiments.items():
        success = run_single_experiment(exp_name, config, base_args)
        results[exp_name] = 'success' if success else 'failed'

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiments')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--quick', action='store_true', help='运行快速测试实验')
    parser.add_argument('--experiment', type=str, default=None, help='运行指定实验')

    args = parser.parse_args()

    if args.experiment:
        if args.experiment in EXPERIMENTS:
            run_single_experiment(args.experiment, EXPERIMENTS[args.experiment], [])
        else:
            print(f"Unknown experiment: {args.experiment}")
    elif args.quick:
        run_quick_experiments(args.output_dir)
    else:
        run_all_experiments(args.output_dir)
