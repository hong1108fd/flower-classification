"""
训练和评估工具模块
"""
import os
import time
import copy
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np


class AverageMeter:
    """计算并存储平均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """计算Top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scheduler: Optional[lr_scheduler._LRScheduler] = None
) -> Dict[str, float]:
    """
    训练一个epoch

    Returns:
        包含loss和accuracy的字典
    """
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 计算准确率
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%'
        })

    if scheduler is not None:
        scheduler.step()

    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = 'Val'
) -> Dict[str, float]:
    """
    评估模型

    Returns:
        包含loss和accuracy的字典
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    pbar = tqdm(dataloader, desc=f'[{split}]')

    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)

        # 计算准确率
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%'
        })

    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


class Trainer:
    """训练器类"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        save_dir: str = './outputs',
        use_swanlab: bool = True,
        swanlab_config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.use_swanlab = use_swanlab

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 初始化swanlab
        if use_swanlab:
            try:
                import swanlab
                self.swanlab = swanlab
                default_config = {
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': train_loader.batch_size,
                    'model': model.__class__.__name__
                }
                if swanlab_config:
                    default_config.update(swanlab_config)

                self.swanlab_run = swanlab.init(
                    project='flower-classification',
                    config=default_config,
                    dir=save_dir
                )
            except ImportError:
                print("Warning: swanlab not installed. Logging will be disabled.")
                self.use_swanlab = False
                self.swanlab = None
        else:
            self.swanlab = None

        # 记录最佳模型
        self.best_acc = 0.0
        self.best_model_weights = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(self, num_epochs: int) -> Dict:
        """完整训练流程"""

        print(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # 训练
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch,
                self.scheduler
            )

            # 验证
            val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device
            )

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['top1_acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['top1_acc'])

            # 打印结果
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Top1 Acc: {train_metrics['top1_acc']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Top1 Acc: {val_metrics['top1_acc']:.2f}%")

            # 记录到swanlab
            if self.use_swanlab:
                self.swanlab.log({
                    'train/loss': train_metrics['loss'],
                    'train/top1_acc': train_metrics['top1_acc'],
                    'train/top5_acc': train_metrics['top5_acc'],
                    'val/loss': val_metrics['loss'],
                    'val/top1_acc': val_metrics['top1_acc'],
                    'val/top5_acc': val_metrics['top5_acc'],
                    'epoch': epoch
                })

            # 保存最佳模型
            if val_metrics['top1_acc'] > self.best_acc:
                self.best_acc = val_metrics['top1_acc']
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"New best model saved! Accuracy: {self.best_acc:.2f}%")

        # 训练结束
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best validation accuracy: {self.best_acc:.2f}%")
        print(f"{'='*60}")

        # 加载最佳模型
        self.model.load_state_dict(self.best_model_weights)

        # 结束swanlab
        if self.use_swanlab:
            self.swanlab.finish()

        return self.history

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'metrics': metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint_latest.pth'))

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))

    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_acc = checkpoint.get('best_acc', 0.0)
        return checkpoint['epoch']


def test_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """
    在测试集上评估模型

    Returns:
        test_loss, test_acc, predictions
    """
    model.eval()
    model = model.to(device)

    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # 计算准确率
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = (all_preds == all_targets).mean() * 100

    return losses.avg, accuracy, all_preds
