"""
模型定义模块
包含 ResNet Baseline, SE-ResNet, CBAM-ResNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """CBAM Channel Attention Module"""

    def __init__(self, channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """CBAM Spatial Attention Module"""

    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResNetBaseline(nn.Module):
    """
    ResNet Baseline 模型
    使用预训练的ResNet，修改最后一层全连接层
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 102,
        pretrained: bool = True
    ):
        super(ResNetBaseline, self).__init__()

        # 加载预训练模型
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name

        # 获取最后一层全连接层的输入特征数
        in_features = self.backbone.fc.in_features

        # 替换最后一层
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # 保存特征提取器
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = self.backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        features = self.features(x)
        return features.view(features.size(0), -1)


class SEResNet(nn.Module):
    """
    SE-ResNet 模型
    在ResNet的基础上添加SE Block
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 102,
        pretrained: bool = True,
        reduction: int = 16
    ):
        super(SEResNet, self).__init__()

        # 加载预训练模型作为基础
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # ResNet18/34 的通道数
            channels = [64, 64, 128, 256, 512]
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name

        # 在每个残差块后添加SE Block
        # 修改layer1, layer2, layer3, layer4
        self._add_se_blocks(channels[1:], reduction)

        # 获取最后一层全连接层的输入特征数
        in_features = self.backbone.fc.in_features

        # 替换最后一层
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # 保存特征提取器
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = self.backbone.fc

    def _add_se_blocks(self, channels: list, reduction: int):
        """在每个layer后添加SE Block"""
        # 对于ResNet18/34，我们需要在layer之后添加SE
        # 这里我们创建一个包装器

        # 原始结构: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # 我们需要在 layer1, layer2, layer3, layer4 之后添加 SE Block

        # 使用Sequential重新构建
        self.backbone.layer1 = nn.Sequential(
            self.backbone.layer1,
            SEBlock(channels[0], reduction)
        )
        self.backbone.layer2 = nn.Sequential(
            self.backbone.layer2,
            SEBlock(channels[1], reduction)
        )
        self.backbone.layer3 = nn.Sequential(
            self.backbone.layer3,
            SEBlock(channels[2], reduction)
        )
        self.backbone.layer4 = nn.Sequential(
            self.backbone.layer4,
            SEBlock(channels[3], reduction)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        features = self.features(x)
        return features.view(features.size(0), -1)


class CBAMResNet(nn.Module):
    """
    CBAM-ResNet 模型
    在ResNet的基础上添加CBAM模块
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        num_classes: int = 102,
        pretrained: bool = True,
        reduction: int = 16
    ):
        super(CBAMResNet, self).__init__()

        # 加载预训练模型作为基础
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name

        # 在每个残差块后添加CBAM
        self._add_cbam_blocks(channels[1:], reduction)

        # 获取最后一层全连接层的输入特征数
        in_features = self.backbone.fc.in_features

        # 替换最后一层
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # 保存特征提取器
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = self.backbone.fc

    def _add_cbam_blocks(self, channels: list, reduction: int):
        """在每个layer后添加CBAM"""
        self.backbone.layer1 = nn.Sequential(
            self.backbone.layer1,
            CBAM(channels[0], reduction)
        )
        self.backbone.layer2 = nn.Sequential(
            self.backbone.layer2,
            CBAM(channels[1], reduction)
        )
        self.backbone.layer3 = nn.Sequential(
            self.backbone.layer3,
            CBAM(channels[2], reduction)
        )
        self.backbone.layer4 = nn.Sequential(
            self.backbone.layer4,
            CBAM(channels[3], reduction)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征向量"""
        features = self.features(x)
        return features.view(features.size(0), -1)


def create_model(
    model_type: str = 'resnet18',
    num_classes: int = 102,
    pretrained: bool = True
) -> nn.Module:
    """
    创建模型的统一接口

    Args:
        model_type: 模型类型，可选 'resnet18', 'resnet34', 'se_resnet18', 'se_resnet34',
                   'cbam_resnet18', 'cbam_resnet34'
        num_classes: 类别数
        pretrained: 是否使用预训练权重

    Returns:
        模型实例
    """
    if model_type in ['resnet18', 'resnet34', 'resnet50']:
        return ResNetBaseline(model_type, num_classes, pretrained)
    elif model_type.startswith('se_'):
        base_model = model_type[3:]  # 移除 'se_' 前缀
        return SEResNet(base_model, num_classes, pretrained)
    elif model_type.startswith('cbam_'):
        base_model = model_type[5:]  # 移除 'cbam_' 前缀
        return CBAMResNet(base_model, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_type in ['resnet18', 'se_resnet18', 'cbam_resnet18']:
        model = create_model(model_type, num_classes=102, pretrained=False)
        model = model.to(device)

        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"{model_type}: output shape = {output.shape}")

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
