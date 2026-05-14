"""
102 Category Flower Dataset 数据加载模块
"""
import os
import scipy.io
import shutil
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class Flowers102Dataset(Dataset):
    """102 Category Flower Dataset"""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        download: bool = True
    ):
        """
        Args:
            root: 数据集根目录
            split: 'train', 'val', 'test'
            transform: 图像变换
            download: 是否自动下载数据集
        """
        self.root = Path(root)
        self.transform = transform
        self.split = split

        # 数据集路径
        self.data_dir = self.root / 'flowers-102'
        self.images_dir = self.data_dir / 'jpg'

        if download:
            self._download_dataset()

        # 加载标签和划分信息
        self._load_data()

    def _download_dataset(self):
        """下载并解压数据集"""
        import tarfile
        import urllib.request

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已下载
        if (self.data_dir / 'jpg').exists():
            print("Dataset already exists, skipping download.")
            return

        print("Downloading 102 Category Flower Dataset...")

        # 下载图片
        images_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
        images_tgz = self.root / '102flowers.tgz'

        if not images_tgz.exists():
            urllib.request.urlretrieve(images_url, images_tgz)

        # 下载标签
        labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
        labels_file = self.data_dir / 'imagelabels.mat'

        if not labels_file.exists():
            urllib.request.urlretrieve(labels_url, labels_file)

        # 下载数据集划分
        split_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'
        split_file = self.data_dir / 'setid.mat'

        if not split_file.exists():
            urllib.request.urlretrieve(split_url, split_file)

        # 解压图片
        print("Extracting images...")
        with tarfile.open(images_tgz, 'r:gz') as tar:
            tar.extractall(path=self.data_dir)

        print("Dataset downloaded and extracted successfully.")

    def _load_data(self):
        """加载数据集标签和划分信息"""
        # 加载标签
        labels_path = self.data_dir / 'imagelabels.mat'
        labels = scipy.io.loadmat(labels_path)['labels'][0]

        # 加载划分索引 (注意：MATLAB索引从1开始)
        split_path = self.data_dir / 'setid.mat'
        split_data = scipy.io.loadmat(split_path)

        # 原始划分：train, test, valid
        # 但我们通常按 train:val:test = 8:1:1 来重新划分
        train_idx = split_data['trnid'][0] - 1  # 转换为0索引
        val_idx = split_data['valid'][0] - 1
        test_idx = split_data['tstid'][0] - 1

        # 根据split选择数据
        if self.split == 'train':
            self.indices = train_idx
        elif self.split == 'val':
            self.indices = val_idx
        elif self.split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"Unknown split: {self.split}")

        self.labels = labels[self.indices]

        # 图片文件名格式: image_00001.jpg, image_00002.jpg, ...
        self.image_files = [f'image_{i+1:05d}.jpg' for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images_dir / self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx]) - 1  # 转换为0-101的标签，确保是int类型
        return image, label

    @property
    def num_classes(self) -> int:
        return 102


def get_data_transforms(input_size: int = 224) -> dict:
    """获取数据增强变换"""

    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证集和测试集变换
    val_transform = transforms.Compose([
        transforms.Resize(int(input_size * 1.143)),  # 256 for 224
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def get_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    获取数据加载器

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    transforms_dict = get_data_transforms(input_size)

    # 创建数据集
    train_dataset = Flowers102Dataset(
        root=root,
        split='train',
        transform=transforms_dict['train'],
        download=True
    )

    val_dataset = Flowers102Dataset(
        root=root,
        split='val',
        transform=transforms_dict['val'],
        download=False
    )

    test_dataset = Flowers102Dataset(
        root=root,
        split='test',
        transform=transforms_dict['test'],
        download=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, 102


if __name__ == '__main__':
    # 测试数据加载
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        root='./data',
        batch_size=32,
        num_workers=0
    )

    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # 测试一个batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
