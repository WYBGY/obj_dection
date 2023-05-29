from typing import List, Tuple, Dict, Optional, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[Any, Tensor]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    """
    anchors生成器，用于在特征图上生成anchors的模块
    :arg
    sizes : Tuple[Tuple[int]]
    aspect_ratios: Tuple[Tuple[float]]
    模块可以计算在每个feature map在多个sizes和ratios下生成的anchors的数量

    注意，所输入的参数sizes和ratios必须长度一致，也就是说，sizes和ratios的每一维对应着在那一个feature map所产生的的anchor的参数
    比如，当有3个feature map的时候，len(sizes) = len(ratios)=3, 就是说每一个sizes对应着3个ratios
    sizes = ((128,), (256,), (512,))
    ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    或者
    sizes=((32, 64, 128, 256, 512),)
    ratios=((0.5, 1.0, 2.0),)


    sizes[i] 和 ratios[i] 也就是在第i个feature map生成anchor的参数可以是不相同的，
    在这个feature map 上每个cell可以生成sizes[i] * ratios[i]个anchors

    """
    __annotations__ ={
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        # 先做输入参数的格式判断，不符合进行转换, (128, 256, 512)转换成((128,), (256, ), (512, ))
        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s, ) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios, ) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        这里的scales和aspect——ratios和init里的一样，感觉没必要再定义一遍
        :param scales: 对应sizes的scale
        :param aspect_ratios: 对用self.aspect_ratios
        :param dtype:
        :param device:
        :return:
        """

        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        # 计算长宽比率，目的是根据面积，求得边长
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1.0/h_ratios
        # 求得各个anchors的宽和高, None的作用是多变出来一维，(10)经过None之后变为(10, 1)
        # 这里求得anchors是不同比率不同尺寸的anchors数量
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        # 求得宽和高后，将anchors都转成以(0,0)为中心的坐标，shape为[len(scales)*len(ratios), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)/2
        # 返回四舍五入的整数坐标点
        return base_anchors.round()

    # 对多个feature map上的anchors的模板都求出来
    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None

            if cell_anchors[0].device == device:
                return

        # 根据sizes和ratios生成anchors模板
        # 对每一个feature map上的anchors模板都求出来，就是对上面的generator anchors作用在每一个feature map的sizes和aspect_ratios
        cell_anchors = [self.generate_anchors(sizes, aspect_ratios, dtype, device)
                        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算在每个feature map上每个cell所产生的anchors的数量
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        计算预测feature map上所有的anchors的坐标
        :param grid_sizes: feature map的height和width
        :param strides: feature map对应着原图的步距
        :return: anchors, 一个batch是一个List，每一个元素是一个image所有feature map的anchors
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        # 对每一个feature map的(height, width)、步距和每一个cell的anchors(以(0,0)为原点的9个anchors)
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # 这里是将feature map上的坐标对应到原图上
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 生成对应的网格坐标，也就是坐标对(x, y)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            # 求得每个cell的坐标，这个坐标并不是真实坐标，而是anchor所对应的偏移量
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # 将anchors加上偏移量，就是anchors在原图上的坐标
            # shifts是cell的数量，形状为(950, 1)，base_anchors为(15, 4), 在相加时采用广播机制，也就是15个anchors，每个都要加上一个偏移量
            shifts_anchors = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            # 将每一张feature map在原图上的anchors放在列表中
            anchors.append(shifts_anchors.reshape(-1, 4))
        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        # 将计算得到的anchors信息进行缓存
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]

        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors

        return anchors

    def forward(self, image_list, feature_maps):
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        """
        根据images和feature maps生成anchors
        :param image_list: 1个batch_size的image
        :param feature_maps: 第0维是feature map的层数，每一维是[batch_size, channels, height, width]
        :return: anchors，表示每张image，每个feature map所生成的anchors的数量
        """
        # 得到feature map的height、width
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        # 图片的尺寸
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        """计算在每个feature map上对应原图上的stride，1个batch上在同一个feature map的尺寸是相同的, image大小也是相同的(因为已经放到同一张画布上了)
           所以strides的形状为[层数，stride_height, stride_width], 
           这也是为什么feature maps为什么是[层数，batch_size, channel, height, width]的原因
           所以这里对于mobilenetV2来说，只有一个feature map,那么strides就是[[height_stride, width_stride]],
           如果是resnet50，strides为[[height_stride1, width_stride1], [height_stride2, width_stride2], .....]"""
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)]
                   for g in grid_sizes]
        # 这里目的是得到self.cell_anchors，也就是每一个feature map上的cell的模板，模板的数量和坐标
        self.set_cell_anchors(dtype, device)
        # 求出每一个feature map上的所有的anchors，anchors的数量为num_cells * len(sizes) * len(ratios)
        """  请注意，这里忽略了batch_size的信息，因为计算时只考虑了stride, 在一个batch的feature map又是一样大小的，
             因此只计算在某个feature map上的cell在原图上生成的anchors的数量"""
        # 并将anchors信息保存在cache中
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        """# 由于上面忽略了batch_size的信息，所求的anchors其实是在某个image所对应的每个feature map上的anchors，所以，对于每张image，
        # 都要复制一遍anchors，因为对于每张image而言，anchors是一样的"""
        # 生成的anchors，每一个元素表示一个image的anchors，这个元素的长度为feature map的数量，表示在每个feature map上的anchors
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 对于一张image而言，将所有的feature map上的anchors进行cat，得到num_feature_maps * num_anchors
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    RPN用于分类和回归的部分，RPN分类只有前景和背景两个类别，回归同样也不分具体类别
    输入是feature maps和anchors，输出是anchors的分类结果和回归参数，是一个多维的，
    [num_feature_map, batch_size, num_anchors, height, width]
    """
    def __init__(self, in_channels, num_anchors):
        """
        :param in_channels: feature map的channel
        :param num_anchors: 每一个cell的anchors的个数，len(sizes) * len(ratios)
        注意，这里num_anchors是根据feature map的层数不同而不同，在mobilenetV2中，num_anchors=15,而在resnet50中，num_anchors=3,
        因为resnet50有5个feature map。
        """
        super(RPNHead, self).__init__()
        # 3*3 卷积
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 预测目标的分数，一共有num_anchors个框,那么分类就是num_anchors个结果
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 回归参数
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        # 输入的feature maps有多层，对每一层都进行分类和回归，这里没有进行flatten操作，只是进行1*1的卷积
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor顺序，并进行reshape
    Args:
        layer: 预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]
    """
    # view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    # view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    # reshape则不需要依赖目标tensor是否在内存中是连续的
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C,  H, W)
    # 调换tensor维度
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    box_cls, box_regression在经过RPNHead之后所得到的shape为[num_feature_maps, batch_size, num_anchors * 1(4), height, width],
    需要将其进行转换，每个feature map转换为[batch_size, num_all_anchors, num_classes or (4)]的形状，然后再flatten操作，
    最终形成shape为[all_num_anchors, 1(4)]
    :param box_cls: 经RNPHead 1*1conv卷积后得到的分类结果
    :param box_regression: 经RPNHead 1*1conv卷积后所得到的回归参数
    :return:
    """

    box_cls_flatten = []
    box_regression_flatten = []
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # N:batch_size, AXC: num_anchors_per_cell * class_num, H: height, W: width
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        # num_anchors
        A = Ax4 // 4
        # class_num
        C = AxC // A
        # premute_and_flatten将[N, AxC, H, W]转成[N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flatten.append(box_cls_per_level)
        # permute_and_flatten将[N, AXC, H., W]转成[N, -1, 4]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flatten.append(box_regression_per_level)
    # 再进行拉平操作，先将每个feature map进行拼接，然后拉平, 拉平后[all_num_anchors_all_feature_map, C(C*4)]
    box_cls = torch.cat(box_cls_flatten, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flatten, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(nn.Module):
    """
    整合整个RPN模块，前面定义了generator anchors、anchors计算和RPNHead部分，该部分将上面的整合，形成整个RPN框架，包括损失计算
    Argument:
        anchor_generator: 用于在feature maps上生成模板anchors的以及将模板anchors对应到图片上的模块；传入的AnchorGenerator的实例化类
        head: 分类和回归的模块；生成objectness 和 regression deltas;
        fg_iou_threshold: 前景的IOU阈值；大于该阈值被当做正例
        bg_iou_threshold: 背景的IOU阈值；小于该阈值被当做反例
        batch_size_per_image: 在训练RPN时，每个bathch_size所选取的框的个数；
        positive_fraction: 正例所占比例；
        pre_nms_top_n：在NMS之前所保留的proposals的个数；
        post_nms_top_n: 在NMS之后所保留的proposals的个数；
        nms_threshold: NMS阈值，
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sample": det_utils.BalancedPositiveNegativeSampler,
        "pre_nms_top_n": Dict[str, int],
        "post_nms_top_n": Dict[str, int]
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_threshold, bg_iou_threshold,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_threshold, score_threshold=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        # 在训练阶段使用
        # 计算anchors与gt的iou值
        self.box_similarity = box_ops.box_iou
        #
        self.proposal_matcher = det_utils.Matcher(fg_iou_threshold, bg_iou_threshold,
                                                  allow_low_quality_matches=True)
        # 正负样本采样
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        # 在testing使用
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_threshold
        self.score_thresh = score_threshold
        self.min_size = 1.

    # 对pre_nms_top_n和post_nms_top_n进行重新定义，分为训练和测试用
    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        """
        获取在每个level上概率排前pre_nms_top_n的anchors
        :param objectness: [batch_size, all_anchors_num]每个anchors的概率
        :param num_anchors_per_level: List, 每个level上的anchors的数量
        :return:
        """
        r = []
        offset = 0
        # 遍历每个level上每个anchor的预测概率
        # 这里将objectness重新按照level划分开，才能遍历，划分开后shape为[batch_size, per_level_num_anchors]
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                # 每一个level的anchors数量
                num_anchors = ob.shape[1]
                # 从定义的参数和num_anchors取最小值，因为有的level可能不足pre_nms_top_n个
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            # 按照objectness中的分数，取出前n个anchors，返回其索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            # 这里要加上offset，因为对于split之后的索引都是从0开始，而一旦拼到一起，就要加上每一层的总数量
            # 0,1,2,3,4,5 ++ 0,1,2,3,4  ++  0,1,2,3,4   ++ .....
            # 0,1,2,3,4,5 ++ (+6)6,7,8,9,10 ++ ()11,12,13,14,15 ++ .....
            r.append(top_n_idx + offset)
            # offset加上num_anchors数量，等于最终拼到一个list的索引
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        对anchors进行处理，剔除小的box，再进行NMS处理，最终得到post_nms_top_n个proposals
        Argument:
            proposals: 所得到的proposals，形状为[batch_size, num_anchors_per_image, 4]
            objectness: 所得到的每个框的分类结果，形状为[batch_size, num_anchors_per_image, 1]
            image_shapes: image的尺寸信息 List[Tuple[int, int]]
            num_anchors_per_level: 每一个level的num_anchors，跟batch_size无关，每个image的每个feature map的anchors数量
        :return:
        """
        # batch_size
        num_images = proposals.shape[0]
        device = proposals.device
        """
        在这段代码中，通过使用detach()函数将 objectness 的梯度流断开，从而防止在后续反向传播中计算其梯度，
        因为这部分梯度不会被用来更新 RPN 的参数。这么做是为了减少计算量，加快反向传播的速度，因为objectness
        的梯度计算会增加计算复杂度。同时，这也有助于提高训练的稳定性，因为 objectness的梯度流被断开后，就不会对后面的计算产生影响。
        """
        objectness = objectness.detach()
        # objectness原本是[batch_size, all_anchors, 1],去掉一个维度
        objectness = objectness.reshape(num_images, -1)
        # levels记录每一个feature map上anchors的索引，第0层的索引全为0，第1层上的anchors索引全为1，如：
        # levels = [(0,0,0,...,0), (1,1,1,...,1), (2,2,2,...,2),...]
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        # 然后cat到一起，levels = [0,0,0...,1,1,1,...,2,2,2,...]
        levels = torch.cat(levels, 0)
        # 然后将levels拓展到与objectness一样的shape，因为现在levels是一张image的levels，每一张都一样，因此相当于levels * batch_size
        levels = levels.reshape(1, -1).expand_as(objectness)
        # 选取每一个level上的概率排前per_nms_top_n的anchors的索引值
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        # 取出找出来的前n大score的objectness, 这里按照每张image，取出对应image中的idx
        objectness = objectness[batch_idx, top_n_idx]
        # 同时取出idx所对应的level
        levels = levels[batch_idx, top_n_idx]
        # 取出对应的proposals
        proposals = proposals[batch_idx, top_n_idx]
        # 将score变成概率，score有正有负
        objectness_prob = torch.sigmoid(objectness)
        #
        final_boxes = []
        final_scores = []
        # 每一张image进行遍历
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 对boxes进行裁减，将超出边界的boxes裁减到边界上
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            # 移除小框，height和width都小于batch_size的框
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # 移除后剩余proposals、scores、level信息
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            # 进行NMS处理, 返回NMS之后的索引
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            # nms处理后，再保留post_nms_top_n个proposals
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算anchors与之最匹配的gt，然后将这些anchors划分为正样本、背景（负样本）和丢弃样本
        Argument：
            :param anchors: List[Tensor]，一个batch_size的image，对应一张image的all_anchors
            :param targets: List[Dict[str, Tensor]]
            :return: 返回每个anchors所匹配到gt_box信息，以及label信息
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张image的anchors和target
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image['boxes']
            # 当图片中没有标注框时, 返回的对应每个anchors的gt为[0,0,0,0]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0], ), dtype=torch.float32, device=device)
            else:
                # 计算anchors与真实的gt的IOU值
                # 一个matrix，iou(gt(i), anchors(j))
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)
                # 找到每个anchors与gt按照iou threshold进行匹配的结果，返回每个anchors所匹配到
                # 小于low_threshold置为-1，between置为-2，允许低质量匹配时，将最大的iou再从-1/-2变回去
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # 将gt_box的四个坐标值对应到对应到所匹配的gt上去，gt_boxes[2, 4], matched_idxs[217413,] -> [217413, 4]
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                # 接下来为每个anchors打上标签标签处理，正样本置为1，负样本置为0, 丢弃样本置为-1
                # 这里在RPN中正样本为1，而在roi—head中，labels就是对应的类别
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)
                # 背景
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0
                # 丢弃样本
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLD
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def compute_loss(self, objectness, pre_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算分类损失和回归损失，先采样，再计算
        Argmuments：
            :param objectness: 模型所预测的anchors所属的类别信息
            :param pre_bbox_deltas: 模型所预测出来的每个anchors的4个回归参数
            :param labels: anchors所对应的gt的类别信息
            :param regression_targets: anchors所对应的gt，所计算出来的回归参数
        :return:
        """
        # 正负样本采样，返回的均为[len(anchors)]大小，对应的位置为1
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将所有image的正样本拼接在一起，找出正样本的索引(等于1的位置)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # 同理，将负样本拼接在一起，找出负样本的索引
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 再将正负样本拼接在一起，拼接后长度应为batch_size_per_image * 2
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # 将objectness展开，也就相当于将objectness中所有的image进行拼接
        objectness = objectness.flatten()
        # 同理labels也是
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框的回归损失，使用smoothL1
        # 计算回归参数时只使用正样本
        box_loss = det_utils.smooth_l1_loss(pre_bbox_deltas[sampled_pos_inds],
                                            regression_targets[sampled_pos_inds],
                                            beta=1/9,
                                            size_average=False) / (sampled_inds.numel())

        # 计算分类损失，RPN中只使用前景和背景
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Arguments:
            images: ImageList格式， 一个batch_size的image
            features: feature maps, OrderDict格式，每个level的feature map
            targets: label数据， 一个batch_size的label, 每个元素是一个Dict，包含boxes信息等
        :return:
            boxes: RPN最终输出的boxes，[List[Tensor]]格式，每个元素是一张image的预测框信息
            losses: Dict[Tensor], 训练时的损失，包括分类损失和回归损失，测试时损失为空。
        """
        # 将每一层的feature map放在一个列表中
        features = list(features.values())

        # 将特征放入head中，生成cls和regression
        # 注意，生成的objectness和regression都是多维的
        objectness, pred_bbox_deltas = self.head(features)

        # 生成anchors信息
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # num_anchors_per_level_shape_tensors是计算每个feature map的anchors的shape,
        # 也就是[num_anchors_per_cell, height, width](除去了batch_size)
        # objectness的shape是[num_features, batch_size, num_anchors*C, height, width]
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # 每个level的anchors的数量，不受batch_size的影响
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        # 将objectness和pre_bbox_deltas转换成[batchsize，all_anchors, C]的形状
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # 根据回归参数和proposals, 将proposals调整到接近Gt
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # proposals的处理，滤除小box，nms处理
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        # 计算RPN的loss
        losses = {}
        if self.training:
            assert targets is not None
            # 将anchors与gt进行匹配，并对每个anchors与gt进行对应，返回对应所对应的坐标和类别
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 根据anchors的坐标和所匹配的gt的坐标，计算回归参数
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)

            # 分别计算RPN损失，包括分类损失和回归损失
            """
            objectness: RPN网络预测的每个anchors的类别信息
            pred_bbox_deltas: RPN网络预测的每个anchors的回归参数信息
            labels: 每个anchors所对应的gt的label(真实标签)
            regression_targets: 每个anchors与真实的gt所计算出来的回归参数
            """
            loss_objectness, loss_rpn_box_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)

            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }

        return boxes, losses






