"""
构建yolov3模型文件，通过读取cfg网络配置文件，构建模型
create_model: 解析cfg文件，构架模型
YOLOLayer: 在predictor后进行后处理
Darknet：构建整个模型
"""
import math

import torch

from build_utils.layers import *
from build_utils.parse_config import *
from build_utils import torch_utils
import torch.nn as nn

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    """
    根据cfg文件的结构，逐层搭建网络模型
    :param modules_defs: 解析出来的cfg文件，列表形式，每个元素是一个block及其参数
    :param img_size:
    :return:
    """
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 先删除model_list中第一个block，net在yolov3中没有使用
    modules_defs.pop(0)
    # 创建一个用于保存上一层输出channel的列表，作为下一层的in_channel, 初始化为3，表示输入尺寸为3
    output_filters = [3]
    # 初始化一个nn.ModuleList类，block放入到这里边
    module_list = nn.ModuleList()
    # 统计和记录哪些特征层的输入将会在后面被使用，用于拼接和残差连接
    routs = []
    yolo_index = -1

    # 遍历列表的block
    for i, mdef in enumerate(modules_defs):
        # 对于每个block创建一个Sequential
        modules = nn.Sequential()

        # 处理convolutional模块
        if mdef["type"] == "convolutional":
            # convolutional层的参数，准备搭建convolutional层
            bn = mdef["batch_normalize"]
            filters = mdef["filters"]
            k = mdef["size"]
            # 有的stride可能不止一个整数，在x，y方向的stride，在这里都是1个int类型
            stride = mdef["stride"] if "stride" in mdef else (mdef["stride_y"], mdef["stride_x"])
            # 判断一下kernel size是不是整数
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k//2 if mdef["pad"] else 0,
                                                       bias=not bn  # 当有bn层时不需要bias
                                                       ))
            else:
                raise TypeError("conv2d filter size must be int type")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果没有bn层，则表示是predictor，后面会被yolo layer使用，这里记录其索引
                routs.append(i)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass
        # 处理maxpool模块
        elif mdef["type"] == "maxpool":
            k = mdef["size"]
            stride=mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k-1)//2)

        # 处理upsample模块
        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2 / 32
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # 处理route模块                                                                  #
        # route模块有两种情况：                                                            #
        # 1、当route为单个值的时候，route则指向那一层的输出                                    #
        # 2、当route是多个值的时候，route则返回这些层输出的concated                            #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        elif mdef["type"] == "route":
            layers = mdef["layers"] # [-2], [-1, -3, -5, -6], [-1, 61]
            # 将这几层的通道数加起来，如果大于0要加上1，由于output_filters记录的是上一层的，小于0则不需要，从后向前数
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # 将用到的这些layer保存起来，如layers=-1,则当前层-1的索引保存，如果是大于0，就把那一层保存，如61
            routs.extend([i + l if l < 0 else l for l in layers])
            # 在channel维度上拼接
            modules = FeatureConcat(layers=layers)
        # 处理shortcut模块, shortcut将上一层的输入与from那一层的进行融合，相加
        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            # 取出上一层的output channels
            filters = output_filters[-1]
            # 将from的那一层的索引保存进routs中，
            routs.append(i + layers[0])
            # 两个层进行相加add
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)
        # 处理yolo模块
        elif mdef["type"] == "yolo":
            # 记录第一个yolo layer, 0, 1, 2
            yolo_index += 1
            # 预测特征图的缩放比例scale
            stride = [32, 16, 8]

            modules = YOLOLayers(anchors=mdef["anchors"][mdef["mask"]],
                                 nc=mdef["classes"],
                                 img_size=img_size,
                                 stride=stride[yolo_index])
            # 初始化权重
            """
            这段代码中的操作是对 YOLO 模型输出的偏置项进行调整，目的是为了更好地适应不同类别和目标的预测。
            具体来说，针对 YOLO 模型中的每个锚框，偏置项的每一行表示了对应锚框预测的参数。在这段代码中，有以下两行操作：
                1. b.data[:, 4] += -4.5：这一行将偏置项中的第 4 列的值减去了 4.5。在 YOLO 模型中，第 4 列对应的是目标置信度
                （objectness confidence）。通过将其减去一个较大的值，例如 4.5，可以将初始的目标置信度设置为较小的值，这有助于模型
                在训练的早期更加关注物体边界框的准确性，促使模型更快地学习到目标的位置和形状。
                2. b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))：这一行将偏置项中的第 5 列及之后的值加上一个常数。
                这些值对应于类别置信度（class confidences）。通过对这些值进行加法操作，可以将初始的类别置信度设置为较小的值。
                具体加的常数为 math.log(0.6 / (modules.nc - 0.99))，其中 0.6 是一个经验值，modules.nc 是类别数。通过加上这个常数，
                可以使模型在初始阶段对各个类别的置信度都接近均匀分布，从而避免初始阶段模型对某些类别的过度偏好。
            """
            try:
                j = -1
                # bias参数形状(255, ), view后(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print("Warning: smart bias initialization failure.", e)

        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # 将每一个block放入到module_list中
        module_list.append(modules)
        # 将上一层的输出放入到列表中, 只有convolutional、concat(route)通道数会发生改变，
        # 其他不变的情况仍然会把上一个block的filters放入(referenced before)
        output_filters.append(filters)

    # 将某一层是否后续被使用进行标记
    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayers(nn.Module):
    """
    对predictor的输出进行处理
    :param
        1、anchors：运用在这个yolo layer的anchors尺寸
        2、nc: num_classes
        3、img_size: 尽在ONNX—EXPORT时使用
        4、stride：预测特征图的缩放比例
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayers, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.stride = stride
        self.na = len(anchors)
        self.nc = nc
        # 每个anchor输出的值的个数
        self.no = nc + 5
        # 初始化网格坐标
        self.nx, self.ny, self.ng = 0, 0, (0, 0)
        # 将anchors缩放到预测特征图上的尺度
        self.anchor_vec = self.anchors/self.stride
        # 更改anchors的视图，原本为2维，改为(batch_size, num_anchors, grid_h, grid_w, wh)
        # 值为1的维度不是固定维度，后续操作根据实际情况进行广播扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        self.grid = None

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # 构建网格的xy坐标,在训练阶段是不需要使用到的，因为训练不需要获得边界框
        if not self.training:
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # 将坐标写成batch_size, num_anchors, grid_h, grid_w, wh的形式，后续batch_size和grid改变可以自动扩充
            self.grid = torch.stack((xv, yv), 2).view(1, 1, self.ny, self.nx, 2).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        if ONNX_EXPORT:
            bs = 1 # batch size
        else:
            # batch_size, predict_param((num_classes + 5) * 3), grid_y, grid_x
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:

                self.create_grids((nx, ny), p.device)
        # 将输入的形状进行更改，更改前(batch_size, 255, 13, 13), view后变为(batch_size, 3, 85, 13, 13)
        # 然后再permute，变为(batch_size, 3, 13, 13, 85)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()

        # 训练模式下直接返回结果，相当于进来只做了一个形状的转换
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            p[:, :2] = (torch.sigmoid(p[:, :2]) + grid) * ng
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p

        else:
            # [bs, na, grid, grid, xywh + obj + nc]
            io = p.clone()
            # 预测的偏移量经过sigmoid函数
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid
            # 预测的wh经过exp后乘以anchors的wh
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            # 将所得的xywh乘以stride换算回原尺度
            io[..., :4] *= self.stride
            # 类别分数部分都经过sigmoid
            torch.sigmoid_(io[..., 4:])
            # 返回(batch_size, num_anchors_all, param_per_anchor)
            return io.view(bs, -1, self.no), p


class Darknet(nn.Module):
    """
    YOLOV3 spp 模型
    :param:
        cfg: 模型的配置文件
        img_size: 图片尺寸，尽在ONNX—EXPORT为True时有用
        verbose: 是否打印模型信息
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络的cfg文件,返回一个列表，每个元素是一个block，包含block的参数信息
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构，逐步搭建模型
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取YOLOLayer的索引
        self.yolo_layers = self.get_yolo_layers()

        self.info(verbose) if not ONNX_EXPORT else None

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        # yolo_out收集每个yolo layer的输出
        # out 收集每个模块的输出
        yolo_out, out = [], []
        if verbose:
            print("0", x.shape)
            str = ""

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__

            if name in ["WeightedFeatureFusion", "FeatureConcat"]:
                if verbose:
                    l = [i-1] + module.layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]
                    str = " >> " + " + ".join(["layer %g %s" % x for x in zip(l, sh)])
                x = module(x, out)
            elif name == "YOLOLayers":
                yolo_out.append(module(x))
            else:
                x = module(x)
            # 只将在后面会使用到的层的输出保存起来
            out.append(x if self.routs[i] else [])

            if verbose:
                print("%g/%g %s -" % (i, len(self.module_list), name), list(x.shape), str)
                str = ""

        # 遍历完所有module后
        if self.training:
            # 训练模式下，将输出进去转个形状就出来了
            return yolo_out

        elif ONNX_EXPORT:
            p = torch.cat(yolo_out, dim=0)
            return p
        else:
            # inference时,yolo模块返回两个值，调整后的结果和原始结果
            x, p = zip(*yolo_out)
            # 将x变成batch_size, num_anchors_all, params
            x = torch.cat(x, 1)
            return x, p

    def info(self, verbose):
        """
        打印模型信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)

    def get_yolo_layers(self):
        """
        获取网络中三个"YOLOLayer"的block的索引
        :return:
        """
        return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == "YOLOLayers"]



