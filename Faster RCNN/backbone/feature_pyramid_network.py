from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    IntermediateLayerGetter 类，它是一个 PyTorch 模型的包装器，可以用于提取模型中间层的特征输出。
    该类的主要作用是在给定的模型中提取指定子模块的特征，并将其按照指定的名称进行输出。
    获取一个Model中你指定要获取的哪些层的输出
    # 这个方法跟create_feature_extractor功能类似，但是这个只能获取一级子模块，无法获得下一级，因此常用create_feature_extractor
    """
    __annotations__ = {
        "return_layers": Dict[str, str]
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 遍历model的子模块，进行正向传播
        # 收集layer1，layer2，layer3，layer4的输出
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FeaturePyramidNetwork(nn.Module):
    """
    FPN网络模块，网络的输入为OrderDict的feature maps
    Argument:
        in_channels_list: 每一个feature map的channel
        out_channels: FPN的channel
        extra_block：同BackboneWithFPN
    """

    def __init__(self, in_channels_list, out_channels, extra_block):
        super(FeaturePyramidNetwork, self).__init__()

        self.inner_blocks = nn.ModuleList()

        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            # 每一个feature map出来后调整一次out_channels
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            # 然后在经过3*3的卷积，形状不变
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
        # 将FPN这几层机型初始化
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_block = extra_block

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        用于获取在 FPN 中某个特定的 inner block 的输出结果。
        :param x:
        :param idx:
        :return:
        """
        num_blocks = len(self.inner_blocks)
        if idx <= 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:

        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks

        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        OrderDict feature maps经FPN计算, FPN将这组特征图上采样到一个共同的分辨率，并对其进行融合，以产生具有不同分辨率的特征图。
        这里对于输入的feature map，前面的（高分辨率的）都融合了其后面的信息，方法是将后面的那一层上采样2倍之后相加。这与yolo中略有不同。
        eg:
            假设输入的return_layers={"0": "layer1", "1": "layer2", "2": "layer3"}，则输出的tensor_fpn一共有3个，
            分别对应layer1、layer2和layer3三个特征层的融合结果。
            具体的融合过程如下，从后向前：
                1、对于layer3特征层，经过self.inner_blocks[2]卷积层进行通道数的调整，得到last_inner。
                2、对于layer3特征层，先将last_inner作为该层对应的预测特征层self.layer_blocks[2]的输入，得到out_layer3。
                3、对于layer2特征层，先通过self.inner_blocks[1]卷积层进行通道数的调整，得到inner_lateral2。然后将last_inner上采样
                   至inner_lateral2的大小，再将其与inner_lateral2相加，得到融合后的特征层top_down_layer2。
                4、对于layer2特征层，将top_down_layer2作为该层对应的预测特征层self.layer_blocks[1]的输入，得到out_layer2。
                5、对于layer1特征层，先通过self.inner_blocks[0]卷积层进行通道数的调整，得到inner_lateral1。然后将top_down_layer2上
                  采样至inner_lateral1的大小，再将其与inner_lateral1相加，得到融合后的特征层top_down_layer1。
                6、对于layer1特征层，将top_down_layer1作为该层对应的预测特征层self.layer_blocks[0]的输入，得到out_layer1。
            最后，输出的tensor_fpn为[out_layer1, out_layer2, out_layer3]，即分别对应融合后的layer1、layer2和layer3
            三个特征层的预测特征矩阵。


            :param x: OrderDict[Tensor]: feature maps
            :return: OrderDict[Tensor],返回的特征图（feature maps）的顺序是从分辨率最高的开始，
                     逐步下采样（resolution decreases）的顺序排列的。
        """
        names = list(x.keys())
        x = list(x.values())

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # 下面是最后一个feature map经过FPN后的计算结果                                                        #
        # # 计算x经过最后layer4经FPN调整out_channels后的结果
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # results保存每个feature map经过FPN后的结果
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))
        # 剩余层的计算过程，从后向前，因为越往后feature map尺度越小，分辨率越小
        # 对于除了最后一层，前面一层都要融合后一层的信息, 因此从后向前
        for idx in range(len(x) - 2, -1, -1):
            # 对于当前层，先对其通道进行调整
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            # 查看feature map的分辨率
            feat_shape = inner_lateral.shape[-2:]
            # 然后将上一层的feature map进行上采样，调整到与当前层尺度相同
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            # 然后进行融合
            last_inner = inner_lateral + inner_top_down
            # 然后得到作为预测的特征层
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 将最后一层的预测特征层的基础上经过最大池化再生成一个feature map
        if self.extra_block is not None:
            results, names = self.extra_block(results, x, names)

        out = OrderedDict([(k, v) for k, v in zip(names, results)])
        return out


class BackboneWithFPN(nn.Module):
    """
    在model的上面加上一个FPN
    使用IntermediaLayerGetter模块（torchvision.models._utils.IntermediateLayerGetter）去提取返回return layers所指定的特征图
    Arguments:
        backbone: 已经经过create_feature_extractor，指定return layers的backbone
        return_layers: Dict[name, new_name]: FPN所采用的那些层，eg:
                        return_layers = {"features.6": "0",   # stride 8
                                         "features.12": "1",  # stride 16
                                         "features.16": "2"}  # stride 32

        in_channels_list: List[int], 每个layer的channel
        out_channels: FPN的channel
        extra_blocks: extra_blocks 是指在构建 Feature Pyramid Network (FPN) 时，除了基础模型（backbone）外，需要添加的
                      额外的上采样模块。在 torchvision 中，这些额外的模块被称为 Extra FPN Blocks，它们用于将来自底层 feature maps
                      的信息提取到更高层次的 feature maps 中，以实现更好的目标检测性能。通常，这些额外模块可以是卷积层或反卷积层等。

    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        """
        当 re_getter 参数设置为 True 时，我们需要为 FPN 创建一个新的 IntermediateLayerGetter 对象来从 backbone 中提取中间层特征图。
        这是因为有些 FPN 模型需要返回 backbone 的多个中间层的特征图，而这些特征图不一定连续，可能是隔了几层才能获得的。因此，需要用 
        IntermediateLayerGetter 来动态地提取这些中间层特征图，并传递给FPN模型使用。
        而当 re_getter 参数设置为 False 时，我们直接将 backbone 传递给 FPN，表示 FPN 不需要重新获取 backbone 的中间层特征图，
        而是直接使用backbone自带的中间层特征图。这种情况一般用于当backbone的输出层与 FPN 所需要的输出层完全一致时。
        
        
        
        """
        # regetter就是对backbone进行重构，前面已经使用create_feature_extractor进行重构了，这里不需要再次重构
        if re_getter is True:
            assert return_layers is not None
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,
                                         out_channels=out_channels,
                                         extra_block=extra_blocks)

        self.out_channels = out_channels

    def forward(self, x):
        # 先x经过backbone
        x = self.body(x)
        # 然后经过FPN输出对应的预测特征层
        x = self.fpn(x)
        return x


class LastLevelMaxPool(nn.Module):
    """
    在最后一层feature map上进行一次最大池化
    输入x是Dict feature maps, 将x的最后一个feature map进行一次最大池化
    """
    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names

