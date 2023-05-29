import os

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 1*1卷积，降维
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_channel)
        # 3*3卷积，shape不变
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # 1*1卷积，升维
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, block_nums, num_classes=1000, include_top=True, norm_layer=None):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64
        # 输出 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # in_channel=64, out_channel=64
        self.layer1 = self._make_layer(block, 64, block_nums[0])
        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        # 步距不为1或者通道数不同时，不能残差叠加，要进行下采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel*block.expansion)
            )
        layers = [block(self.in_channel, channel, downsample=downsample, stride=stride, norm_layer=norm_layer)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    搭建resnet_fpn backbone
        :param pretrain_path: 预训练模型权重路径
        :param norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
                           如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
                           (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        :param trainable_layers: 指定训练哪些层结构
        :param returned_layers: 指定哪些层的输出需要返回
        :param extra_blocks: 在输出的预测特征层基础上额外添加的层结构
        :return:
    """
    # 原始resnet backbone
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False, norm_layer=norm_layer)
    # frozen层参数
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist".format(pretrain_path)
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))
    # 需要训练的层，[layer4, layer3, layer2]
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # 如果全训练的话，要加上bn1层
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # 只训练不在layers_to_train列表中的参数
    for name, parameter in resnet_backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    assert min(returned_layers) > 0 and max(returned_layers) < 5

    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    # in_channel是backbone的输出的channel，in_channel在resnet中不断变化，最初64 -> 64*4 -> 512 -> 512*4
    in_channels_stage2 = resnet_backbone.in_channel // 8
    # 每个预测特征层的channel的个数
    in_channels_list = [in_channels_stage2 * 2 ** (i-1) for i in returned_layers]
    # 通过FPN后的out_channel个数
    out_channels = 256
    # 构建带有FPN的backbone
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
