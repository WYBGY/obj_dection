from torch import nn
import torch
from torchvision.ops import misc


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    保证所有层的channel都是8的整数倍
    :param ch:
    :param divisor:
    :param min_ch:
    :return:
    """
    if min_ch is None:
        min_ch = divisor

    new_ch = max(min_ch, int(ch + divisor/2) // divisor * divisor)
    # 不能过小
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 定义一个继承nn.Sequential的类，集成了Conv、BatchNorm、Relu
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kenel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kenel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kenel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_channel),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        self.use_shortcut = stride == 1 and in_channel == out_channel
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channel, hidden_channel, kenel_size=1, norm_layer=norm_layer))

        layers.extend([
            # 3*3 deepwise conv, 对每一个feature map运用卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel, norm_layer=norm_layer),
            # 1*1 pointwise conv
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            norm_layer(out_channel)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, weight_path=None, norm_layer=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32*alpha, round_nearest)
        last_channel = _make_divisible(1280*alpha, round_nearest)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        """
        inverted_residual_setting 是一个列表，其中包含 MobileNetV2 中每个 block 的配置信息。每个block配置信息由以下几个元素组成：
            input_channels：该 block 输入的通道数
            output_channels：该 block 输出的通道数
            kernel_size：该 block 使用的卷积核大小
            stride：该 block 使用的步长大小
            expand_ratio：该 block 中 expand 卷积层输出通道数相对于输入通道数的倍数
            use_se：该 block 中是否使用 Squeeze-and-Excitation 模块
        """
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = []
        # 开始对图像经过3*3的卷积，输出channel为3,
        features.append(ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer))
        # 一共7个block，参数为上面定义的
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c*alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, last_channel, 1, norm_layer=norm_layer))

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(last_channel, num_classes))

        if weight_path is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
        else:
            self.load_state_dict(torch.load(weight_path))


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



