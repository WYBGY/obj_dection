import torch
import math
from typing import List, Tuple
from torch import Tensor


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    reference_boxes和proposals均是是[xmin, ymin, xmax, ymax]的坐标
    转换的主函数，按照ppt中的方法进行转换, x,y为GT的中心点坐标，a:表示anchor，w, h为width和height
    dx = (x - xa) / wa
    dy = (y - ya) / ha
    dw = log(w/wa)
    dh = log(h/ha)
    """
    # weights
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    # proposals的unsqueeze
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # reference_box的unsqueeze
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # 计算wa和ha
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # 计算中心点坐标
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights
    # reference—boxes也是一样的
    gt_widths = reference_boxes_x2 - reference_boxes_x1


    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # 按照公式计算dx, dy, dw, dh
    target_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    target_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    target_dw = ww * torch.log(gt_widths / ex_widths)
    target_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((target_dx, target_dy, target_dw, target_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    BBOX和回归参数之间的相互转换
        encode: 根据anchors/proposals和groundtruth来计算回归参数
        decode: 根据回归参数和groundtruth，将proposals调整成为BBOX
    """
    def __init__(self, weights, bbox_xform_clip=math.log(1000./16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        :param weights:通过乘以权重，可以对不同维度的偏移量进行缩放，使得它们对应的物理意义更加平衡，以便更好地进行训练和推理
        :param bbox_xform_clip: 确保proposals经过回归参数调整后，不会超出图像边界的裁减参数
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        encode是根据GT和proposals获得回归参数，计算方法在ppt中

        :param reference_boxes: List[Tensor], 每个proposals/anchor对应的groundTruth，因此其长度应该与proposals长度一致
        :param proposals: List[Tensor] proposals
        :return: 回归参数
        """
        # 每张图片上的proposals的boxes数量，reference_boxes显然是经过处理的，与proposals的数量相同
        boxes_per_image = [len(b) for b in reference_boxes]
        # 将一个batch的所有box合并放在一起，便于后面的计算
        reference_boxes = torch.cat(reference_boxes, dim=0)
        # proposals也一样
        proposals = torch.cat(proposals, dim=0)

        # 计算回归参数dx, dy, dw, dh
        targets = self.encode_signal(reference_boxes, proposals)
        # 最后返回时，再按照batch_size分开
        return targets.split(boxes_per_image, 0)

    def encode_signal(self, reference_boxes, proposals):
        # 进行转换，主要是对dtype、device进行统一
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """
        根据回归参数， 将proposals调整到接近GT
        Args:
            rel_codes: 回归参数
            boxes： proposals/anchors, 每张image的所有的anchors/proposals
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # 每个image的proposals的数量
        boxes_per_image = [b.size(0) for b in boxes]
        # 将所有image的proposals放在一起
        concat_boxes = torch.cat(boxes, dim=0)
        # 计算一个batch的所有的box
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        #
        pred_boxes = self.decode_signal(rel_codes, concat_boxes)

        # 防止pred_boxes为空时报错
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_signal(self, rel_codes, boxes):
        """
        根据回归参数，将proposals转换成接近Gt的pred_boxes,按照转换公式：
        tx(*)，ty(*), th(*), tw(*) 为回归参数， x,y,w,h为proposals的参数
        tx(*) = (x* - xa) / wa  →→→  x* = [tx(*) * wa] + xa
        ty(*) = (y* - ya) / ha  →→→  y* = [ty(*) * ha] + ya
        tw(*) = log(w* / wa)    →→→  w* = exp(tw(*)) * wa
        th(*) = log(h* / ha)    →→→  h* = exp(th(*)) * ha
        """

        boxes = boxes.to(rel_codes.dtype)
        #
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights # RPN中weights为[1, 1, 1, 1],Faster RCNN中为[10, 10, 5, 5]
        # 回归参数dx, dy, dw, dh，除以对应的权重
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # 限制一下dw和dh，因为dw和dh要进入exp函数，防止过大
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 计算pred—boxes，按照公式, None是为了拓展出来一个维度从(n)变为(n, 1)能够计算
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 根据中心点坐标和h、w计算box的xmin, ymin, xmax, ymax
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    # 小于iou阈值的索引置位-1， 在阈值之间的设为-2
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLD = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLD": int
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        将proposals与anchors进行匹配，便于进行正负样本划分
        :param high_threshold: 大于该threshold，会被划定为候选框
        :param low_threshold: threshold下限，小于该阈值被划定为负样本
        :param allow_low_quality_matches: 是否允许低质量匹配，若为True, 就启用最大iou那个匹配
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLD = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        计算每个anchors与gt匹配的最大的iou值，并记录索引
        :param match_quality_matrix: MXN tensor， M：ground—truth num，N： anchors num
        :return: N[i], anchors所匹配到的gt，i是gt的索引，为-1或-2时表示没有匹配到或者丢弃
        """
        # 先看下match_quality_matrix是否为空,gt是否为空和proposals是否为空
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No grounding Truth for one of images during training")
            else:
                raise ValueError("No proposals boxes available for one of images during training")

        # 开始匹配 M(gt) * N(proposals)
        # 对于每一个proposal，找出与之匹配的最大gt，也就是按列取最大值,dim=0
        # matches_vals表示ious, matches表示anchors的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None
        # iou小于阈值的索引，[False, True, False, False, ....]
        below_low_threshold = matched_vals < self.low_threshold
        # iou在low_threshold和high_threshold之间的
        between_threshold = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        # 将小于low_threshold的索引对应的索引标为-1，注意这里是将anchors的索引标记为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        # 将在low_threshold和high_threshold之间的索引置为-2
        matches[between_threshold] = self.BETWEEN_THRESHOLD
        # 当设置了低质量匹配，需要将那部分即使不满足大于iou的再变回来，选取最大的iou的变回来
        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        当没有大于iou大于high_threshold时，则启用低质量匹配，即找到iou最大的anchors
        :param matches: 已经置为-1和-2的gt的索引
        :param all_matches: 匹配后的anchors对应的gt的索引，也就是没有置位-1和-2的索引
        :param match_quality_matrix:
        :return:
        """
        # 找出与gt所匹配到的最大的iou值，M*N，也就是按行取最大值,返回最大的值，
        # 因为1个gt可能匹配到多个anchors，但是这里不能返回多个，所以到下一步
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # 找到这些索引，可能有1个gt可能匹配到多个最大的iou anchors，
        # 找到对应的索引，(tensor1, tensor2), tensor1是gt的索引，tensor2是anchors的索引
        gt_pred_pairs_of_highest_quality = torch.where(torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None]))
        # 取出最大值对应的索引
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 将之前已经置为-1和-2的那部分即使是below和between的再变回来, 变回匹配到最大值的那个gt
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class BalancedPositiveNegativeSampler(object):
    """
    正负样本采样，适用于RPN和整个网络的采样
    """
    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            :param batch_size_per_image: 训练时每张图片所采集的anchors样本数量
            :param positive_fraction: 正样本所占比例
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        正负样本的采样，输入是每个anchors所匹配到的GTbox的索引，并且经过iou阈值处理后的，正样本为类别编号，负样本为0，丢弃样本-1
        :param matched_idxs: 如上述，一个batch所有图片所匹配的gtbox的索引
        :return: 正样本pos_idx (list[tensor]) 负样本neg_idx (list[tensor])
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张image
        for matched_idxs_per_image in matched_idxs:
            # 获取正样本的索引，正样本为>=1，在RPN中正样本为1，在整个分类中，正样本为1~20
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # 负样本=0,
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 所指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # 实际正样本与指定的正样本，实际正样本一般比较少
            num_pos = min(positive.numel(), num_pos)
            # 指定负样本的数量， 正样本不够的话全部由负样本填充
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            # 正负样本数量确定后，随机从正负样本中采样
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]
            # 为正负样本创建0,1的模板，大小都等于anchors的总数量
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            # 正负样本分开保存的，因此，在对应位置都置为1
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        return pos_idx, neg_idx


def smooth_l1_loss(input, target, beta: float = 1./9, size_average: bool=True):
    """
    计算边界框回归损失
    smooth_l1_loss = 0.5 * x^2 if |x| < 1, else |x| - 0.5
    :param input: 预测的边界框所计算出来的回归参数
    :param target: 真实的回归参数
    :param beta: 用于平滑，当预测值与真实值小于beta时采用平方项，当差距较大时使用L1函数
    :param size_average: 用于控制返回的损失是否取平均值
    :return:
    """
    # 计算预测值与目标值的差距
    n = torch.abs(input - target)
    # 判断n与beta的关系，决定用平方还是L1
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n**2/beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
