# 🌸 102 Category Flower Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 PyTorch 的花朵分类项目，使用 ImageNet 预训练的卷积神经网络进行迁移学习，实现 Oxford 102 类花朵数据集上的高精度识别。

## 📋 项目简介

本项目在 Oxford 102 Category Flower Dataset 上进行了系统性的深度学习实验：

| 实验类型 | 描述 |
|---------|------|
| **Baseline** | 使用预训练 ResNet-18/34，修改输出层进行微调 |
| **超参数分析** | 探究学习率、batch size、训练轮数对性能的影响 |
| **预训练消融** | 对比预训练模型 vs 从零训练模型的性能差异 |
| **注意力机制** | 在 ResNet 基础上添加 SE-Block 和 CBAM 模块 |

### 实验结果

| 模型 | 预训练 | 验证集准确率 | 测试集准确率 |
|------|:------:|:------------:|:------------:|
| ResNet-18 | ✓ | 88.63% | **84.55%** |
| ResNet-18 | ✗ | 47.75% | 42.09% |
| SE-ResNet-18 | ✓ | 88.82% | 85.53% |
| **CBAM-ResNet-18** | ✓ | **89.02%** | **85.93%** |

> 💡 **关键发现**: 预训练权重带来 **42.46%** 的准确率提升；CBAM 注意力机制达到最佳性能。

---

## 🛠️ 环境配置

### 方式一：使用 Conda（推荐）

```bash
# 1. 创建虚拟环境
conda create -n flower_cls python=3.10 -y
conda activate flower_cls

# 2. 安装 PyTorch (根据你的 CUDA 版本选择)
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 方式二：使用 pip + venv

```bash
# 1. 创建虚拟环境
python -m venv flower_cls_env

# Windows 激活
flower_cls_env\Scripts\activate

# Linux/Mac 激活
source flower_cls_env/bin/activate

# 2. 安装依赖
pip install -r requirements.txt
```

### 主要依赖

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
Pillow >= 9.0.0
tqdm >= 4.65.0
matplotlib >= 3.7.0
scikit-learn >= 1.2.0
swanlab >= 0.3.0
seaborn >= 0.12.0
```

---

## 📦 数据集准备

### 方式一：自动下载（推荐）

首次运行训练脚本时，程序会自动下载 Oxford 102 Category Flower Dataset 到 `./data` 目录：

```bash
python train.py --model resnet18 --pretrained
```

### 方式二：手动下载

如果自动下载失败，请手动下载以下文件：

| 文件 | 大小 | 下载链接 |
|------|------|----------|
| 102flowers.tgz | ~330MB | [下载](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) |
| imagelabels.mat | ~4KB | [下载](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat) |
| setid.mat | ~3KB | [下载](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat) |

下载后放置到以下目录结构：

```
data/
└── flowers-102/
    ├── jpg/           # 解压 102flowers.tgz 得到
    │   ├── image_00001.jpg
    │   ├── image_00002.jpg
    │   └── ...
    ├── imagelabels.mat
    └── setid.mat
```

---

## 🚀 快速开始

### 1. 训练 Baseline 模型

```bash
# 使用默认参数训练 ResNet-18
python train.py --model resnet18 --pretrained --epochs 50 --batch_size 32 --lr 0.01

# 训练 ResNet-34
python train.py --model resnet34 --pretrained --epochs 50

# 不使用预训练（从零训练）
python train.py --model resnet18 --no-pretrained --epochs 50
```

### 2. 运行注意力机制模型

```bash
# SE-ResNet-18
python train.py --model se_resnet18 --pretrained --epochs 50

# CBAM-ResNet-18
python train.py --model cbam_resnet18 --pretrained --epochs 50
```

### 3. 批量运行所有实验

```bash
# 运行完整实验（耗时较长，约数小时）
python run_experiments.py

# 运行快速测试（每个模型仅训练 10 epochs）
python run_experiments.py --quick

# 运行特定实验
python run_experiments.py --experiment baseline_resnet18
```

### 4. 运行特定实验模式

```bash
# 超参数分析
python train.py --mode hyperparameter

# 预训练消融实验
python train.py --mode ablation

# 注意力机制对比
python train.py --mode attention

# 运行所有实验
python train.py --mode all
```

---

## 📊 测试与评估

### 测试单个模型

```bash
# 使用训练好的模型进行测试
python train.py --model resnet18 --pretrained --mode train --epochs 0
```

### 模型权重

训练好的模型权重保存在 `./outputs/` 目录：

```
outputs/
├── resnet18_pretrainedTrue_lr0.01_bs32/
│   ├── best_model.pth        # 最佳模型权重
│   ├── checkpoint_latest.pth # 最新检查点
│   ├── config.json           # 训练配置
│   └── results.json          # 训练结果
├── se_resnet18_.../
├── cbam_resnet18_.../
└── ...
```

### 加载预训练权重进行推理

```python
import torch
from models.resnet import create_model

# 创建模型
model = create_model('resnet18', num_classes=102, pretrained=False)

# 加载权重
model.load_state_dict(torch.load('outputs/xxx/best_model.pth'))
model.eval()

# 推理
with torch.no_grad():
    output = model(input_image)
    pred = output.argmax(dim=1)
```

---

## 📈 SwanLab 可视化

本项目使用 [SwanLab](https://swanlab.cn/) 进行训练可视化。

### 查看 SwanLab 日志

```bash
# 训练时会自动上传日志到 SwanLab
python train.py --model resnet18 --pretrained --use_swanlab

# 训练结束后，终端会显示 SwanLab 链接
# 点击链接即可在网页端查看训练曲线
```

### 可视化内容

- 📉 训练/验证 Loss 曲线
- 📈 训练/验证准确率曲线
- 🎯 Top-5 准确率
- ⏱️ 学习率变化曲线

---

## 📁 项目结构

```
flower_classification/
├── data/
│   ├── __init__.py
│   └── dataset.py              # 数据加载和预处理
├── models/
│   ├── __init__.py
│   └── resnet.py               # 模型定义 (ResNet, SE-ResNet, CBAM-ResNet)
├── utils/
│   ├── __init__.py
│   ├── trainer.py              # 训练和评估工具
│   └── visualization.py        # 可视化工具
├── outputs/                    # 模型权重输出目录
├── figures/                    # 可视化图片输出目录
├── logs/                       # 训练日志目录
├── train.py                    # 主训练脚本
├── run_experiments.py          # 批量实验脚本
├── requirements.txt            # 依赖列表
└── README.md
```

---

## ⚙️ 训练参数说明

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | resnet18 | 模型类型: resnet18/34, se_resnet18/34, cbam_resnet18/34 |
| `--pretrained` | True | 使用 ImageNet 预训练权重 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 批量大小 |
| `--lr` | 0.01 | 学习率 |
| `--lr_mult` | 0.1 | 预训练层学习率倍数 |
| `--scheduler` | cosine | 学习率调度器: step/cosine/multistep/none |
| `--device` | cuda | 训练设备: cuda/cpu |

### 完整参数列表

```bash
python train.py --help
```

---

## 🔬 模型架构

### Baseline: ResNet-18/34

```
Input (224×224×3)
    ↓
Conv1 + MaxPool
    ↓
┌─────────────────┐
│ ResNet Stages   │  ← ImageNet 预训练权重
│ (Layer1-Layer4) │    使用较小学习率微调
└─────────────────┘
    ↓
AdaptiveAvgPool
    ↓
┌─────────────────┐
│ FC (512 → 102)  │  ← 新添加的分类层
└─────────────────┘    使用较大学习率训练
    ↓
Output (102 classes)
```

### SE-ResNet

在 ResNet 每个 stage 后添加 Squeeze-and-Excitation Block：

```
Feature Maps (H×W×C)
    ↓
Global Average Pooling → (1×1×C)
    ↓
FC → ReLU → FC → Sigmoid
    ↓
Channel-wise Multiplication
    ↓
Re-calibrated Features
```

### CBAM-ResNet

在 ResNet 每个 stage 后添加 Convolutional Block Attention Module：

```
Feature Maps
    ↓
┌─────────────────────────┐
│ Channel Attention       │  AvgPool & MaxPool → MLP → Sigmoid
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Spatial Attention       │  Channel Pool → Conv → Sigmoid
└─────────────────────────┘
    ↓
Refined Features
```

---

## 📝 引用

如果使用 Oxford 102 Category Flower Dataset，请引用：

```bibtex
@InProceedings{Nilsback08,
   author = "Nilsback, M-E. and Zisserman, A.",
   title = "Automated Flower Classification over a Large Number of Classes",
   booktitle = "Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing",
   year = "2008"
}
```

---

## 📄 License

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
