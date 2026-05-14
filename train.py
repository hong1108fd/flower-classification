"""
主训练脚本
支持不同模型和实验配置
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from data.dataset import get_dataloaders
from models.resnet import create_model
from utils.trainer import Trainer, test_model


def get_parameter_groups(model: nn.Module, lr: float, weight_decay: float, lr_mult: float = 0.1):
    """
    获取参数组，对预训练层使用较小的学习率

    Args:
        model: 模型
        lr: 基础学习率
        weight_decay: 权重衰减
        lr_mult: 预训练层学习率倍数

    Returns:
        参数组列表
    """
    # 获取分类器参数名
    classifier_names = ['fc.weight', 'fc.bias', 'classifier.weight', 'classifier.bias']

    # 分组参数
    pretrained_params = []
    new_params = []

    for name, param in model.named_parameters():
        if any(n in name for n in classifier_names):
            new_params.append(param)
        else:
            pretrained_params.append(param)

    return [
        {'params': pretrained_params, 'lr': lr * lr_mult},
        {'params': new_params, 'lr': lr}
    ], weight_decay


def train_baseline(args):
    """训练Baseline模型"""
    print("=" * 60)
    print("Training Baseline Model")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据加载器
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size
    )

    print(f"Dataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    # 创建模型
    model = create_model(
        model_type=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained
    )

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器 - 对预训练层使用较小学习率
    if args.pretrained and args.different_lr:
        param_groups, wd = get_parameter_groups(
            model, args.lr, args.weight_decay, args.lr_mult
        )
        optimizer = optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=wd,
            nesterov=args.nesterov
        )
        print(f"Using different learning rates: pretrained={args.lr * args.lr_mult}, new={args.lr}")
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )

    # 学习率调度器
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=args.gamma)
    else:
        scheduler = None

    # 实验名称
    exp_name = f"{args.model}_pretrained{args.pretrained}_lr{args.lr}_bs{args.batch_size}"
    if args.exp_name:
        exp_name = args.exp_name

    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    config = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # SwanLab配置
    swanlab_config = {
        'model': args.model,
        'pretrained': args.pretrained,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'optimizer': 'SGD',
        'scheduler': args.scheduler
    }

    # 训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        use_swanlab=args.use_swanlab,
        swanlab_config=swanlab_config
    )

    # 训练
    history = trainer.train(args.epochs)

    # 测试
    print("\nEvaluating on test set...")
    test_loss, test_acc, _ = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 保存结果
    results = {
        'history': history,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': trainer.best_acc
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_dir}")
    return results


def run_hyperparameter_search(args):
    """超参数搜索"""
    print("=" * 60)
    print("Running Hyperparameter Search")
    print("=" * 60)

    # 定义搜索空间
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    epochs_list = [30, 50, 100]

    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            for epochs in epochs_list:
                print(f"\n\nTraining with lr={lr}, batch_size={bs}, epochs={epochs}")

                # 更新参数
                args.lr = lr
                args.batch_size = bs
                args.epochs = epochs
                args.exp_name = f"hp_search_lr{lr}_bs{bs}_ep{epochs}"

                # 训练
                result = train_baseline(args)
                results.append({
                    'lr': lr,
                    'batch_size': bs,
                    'epochs': epochs,
                    'best_val_acc': result['best_val_accuracy'],
                    'test_acc': result['test_accuracy']
                })

    # 保存结果
    with open(os.path.join(args.output_dir, 'hyperparameter_search_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_ablation_study(args):
    """预训练消融实验"""
    print("=" * 60)
    print("Running Pretrained Ablation Study")
    print("=" * 60)

    results = {}

    # 1. 使用预训练
    print("\n\n[1] Training with Pretrained Weights")
    args.pretrained = True
    args.exp_name = "ablation_pretrained"
    pretrained_results = train_baseline(args)
    results['pretrained'] = pretrained_results

    # 2. 不使用预训练
    print("\n\n[2] Training from Scratch")
    args.pretrained = False
    args.exp_name = "ablation_scratch"
    scratch_results = train_baseline(args)
    results['scratch'] = scratch_results

    # 对比结果
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    print(f"Pretrained: Val Acc = {results['pretrained']['best_val_accuracy']:.2f}%, "
          f"Test Acc = {results['pretrained']['test_accuracy']:.2f}%")
    print(f"From Scratch: Val Acc = {results['scratch']['best_val_accuracy']:.2f}%, "
          f"Test Acc = {results['scratch']['test_accuracy']:.2f}%")
    print(f"Improvement: {results['pretrained']['test_accuracy'] - results['scratch']['test_accuracy']:.2f}%")

    # 保存结果
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_attention_experiments(args):
    """注意力机制实验"""
    print("=" * 60)
    print("Running Attention Mechanism Experiments")
    print("=" * 60)

    models_to_test = ['resnet18', 'se_resnet18', 'cbam_resnet18']
    results = {}

    for model_type in models_to_test:
        print(f"\n\nTraining {model_type}")
        args.model = model_type
        args.exp_name = f"attention_{model_type}"
        model_results = train_baseline(args)
        results[model_type] = model_results

    # 对比结果
    print("\n" + "=" * 60)
    print("Attention Mechanism Results")
    print("=" * 60)
    for model_type, res in results.items():
        print(f"{model_type}: Val Acc = {res['best_val_accuracy']:.2f}%, "
              f"Test Acc = {res['test_accuracy']:.2f}%")

    # 保存结果
    with open(os.path.join(args.output_dir, 'attention_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description='Flower Classification Training')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data', help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')

    # 模型参数
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'se_resnet18', 'se_resnet34',
                                 'cbam_resnet18', 'cbam_resnet34'],
                        help='模型类型')
    parser.add_argument('--pretrained', action='store_true', default=True, help='使用预训练权重')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='不使用预训练权重')
    parser.add_argument('--input_size', type=int, default=224, help='输入图像大小')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--lr_mult', type=float, default=0.1, help='预训练层学习率倍数')
    parser.add_argument('--different_lr', action='store_true', default=True,
                        help='对预训练层使用不同学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--nesterov', action='store_true', default=True, help='使用Nesterov动量')

    # 学习率调度
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'multistep', 'none'],
                        help='学习率调度器')
    parser.add_argument('--step_size', type=int, default=30, help='StepLR步长')
    parser.add_argument('--gamma', type=float, default=0.1, help='学习率衰减因子')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--use_swanlab', action='store_true', default=True, help='使用SwanLab记录')

    # 实验类型
    parser.add_argument('--exp_name', type=str, default='', help='实验名称')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'hyperparameter', 'ablation', 'attention', 'all'],
                        help='实验模式')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行实验
    if args.mode == 'train':
        train_baseline(args)
    elif args.mode == 'hyperparameter':
        run_hyperparameter_search(args)
    elif args.mode == 'ablation':
        run_ablation_study(args)
    elif args.mode == 'attention':
        run_attention_experiments(args)
    elif args.mode == 'all':
        print("Running all experiments...")
        run_hyperparameter_search(args)
        run_ablation_study(args)
        run_attention_experiments(args)


if __name__ == '__main__':
    main()
