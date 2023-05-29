import glob
import math
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from build_utils import torch_utils

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": '{:11.5g}'.format})
matplotlib.rc("font", **{"size": 11})

cv2.setNumThreads(0)


def check_file(file):
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob("./**/" + file, recursive=True)
        assert len(file), "file Not Found: %s"%file
        return files[0]


def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2 # xcenter
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2 # ycenter
    y[:, 2] = x[:, 2] - x[:, 0] # width
    y[:, 3] = x[:, 3] - x[:, 1] # height
    return y


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测框转换回原图的尺寸， img1_shape是现在的尺度，img0_shape是要换算回的尺度

    :param img1_shape: 缩放后的尺度，也就是转成原图的尺度
    :param coords: 预测的坐标信息(xywh)
    :param img0_shape: 缩放前的尺度
    :param ratio_pad: 缩放比例和pad
    :return:
    """
    # 当ratio_pad为空时，需要计算ratio和pad
    if ratio_pad is None:
        gain = max(img1_shape) / max(img0_shape)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])


def wh_iou(wh1, wh2):
    """
    计算wh1和wh2的iou, 只根据各自的长和宽计算
    :param wh1:
    :param wh2:
    :return:
    """
    wh1 = wh1[:, None] # [N, 1, 2]
    wh2 = wh2[None] # [1, M, 2]
    inter = torch.min(wh1, wh2).prod(2) # [N, M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def smooth_BCE(eps=0.01):
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    """
    Focal Loss 是一种用于处理类别不平衡和困难样本的损失函数，特别适用于目标检测和图像分类任务。它的作用主要体现在以下几个方面：
        1、类别不平衡处理：在训练数据中，某些类别可能具有较少的样本数量，导致模型对这些类别的学习效果较差。Focal Loss 通过调整样本的权重，
           使得模型更加关注难以分类的样本和少数类别样本，从而缓解类别不平衡问题。

        2、困难样本处理：在训练过程中，存在一些难以分类的样本，它们可能具有较高的误分类概率，而传统的损失函数难以有效地处理这些样本。
           Focal Loss 通过引入一个调整因子，对误分类的样本施加较大的损失权重，从而更加关注这些困难样本，促使模型更好地学习它们。

        3、提高模型鲁棒性：Focal Loss 能够减轻易分样本对模型训练的影响，使得模型更加鲁棒。通过减少易分样本的损失权重，Focal Loss
           能够使模型更加专注于难以分类的样本和少数类别样本，提高模型在这些样本上的性能。

    FL(pt) = -αt*(1-pt)^γ*log(pt)
    pt = p if y = 1 else 1-p
    αt = α if y = 1 else 1-α
    """

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.0000001 - p_t) ** self.gamma

        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def box_iou(box1, box2):
    """
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):

    box2 = box2.t()
    #
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # box1的wh
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    # box2的wh
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    # union area
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)

        if GIoU:
            c_area = cw * ch + 1e-16
            return iou - (c_area - union) / c_area
        if DIoU or CIoU:
            c2 = cw ** 2 + ch ** 2 + 1e-16
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / cv2 + v * alpha)
    return iou


def compute_loss(p, targets, model):
    device = p[0].device
    # 分类损失
    lcls = torch.zeros(1, device=device) # Tensor(0)
    # 定位损失
    lbox = torch.zeros(1, device=device) # Tensor(0)
    # 置信度损失
    lobj = torch.zeros(1, device=device) # Tensor(0)
    # 根据anchors和gtbox，将anchors与gt进行样本匹配，返回匹配到的样本的类别、坐标参数（真实值）、索引信息、anchor尺度信息
    tcls, tbox, indices, anchors = build_targets(p, targets, model)

    h = model.hyp
    # loss reduction sum or mean
    red = "mean"

    # 定义损失函数
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device), reduction=red)

    # class label smoothing,这里设置的eps=0.0,不起作用,cp=1, cn=0
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss, 使用focal loss处理难分样本和样本不均衡等问题
    g = h["fl_gamma"]
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    # 遍历3个yolo layer
    for i, pi in enumerate(p):
        # pi -> [batch_size, num_anchors, grid_x, grid_y, params_per_anchor]
        # indices -> [image_index, anchors_index(0, 0, 1, 1, 2....), grid_cell_y, grid_cell_x]
        b, a, gj, gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0], device=device)
        # 所匹配到的样本的数量
        nb = b.shape[0]
        if nb:
            # 所匹配到的样本预测信息, 根据各个索引信息(image_idx, anchor_idx, grid_y, grid_x),十分巧妙
            # 获取预测结果，ps -> [num_examples, 25]
            ps = pi[b, a, gj, gi]

            # GIOU
            # 前两个值为x, y
            pxy = ps[:, :2].sigmoid()
            # 第2~4为w h
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            # 拼接在一起，pbox为模型预测回归参数
            pbox = torch.cat((pxy, pwh), 1)
            # 计算iou，tbox是真实边界参数
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)
            # giou损失
            lbox += (1 - giou).mean()

            # 匹配到样本的objectness, 直接等于giou了
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)

            # 分类损失
            if model.nc > 1:
                # 定一个[num_examples, num_classes]的矩阵，负样本为0， cn=0
                t = torch.full_like(ps[:, 5:], cn, device=device)
                # 第i个特征层上，targets的类别，0~num_examples上，所属哪个类别，哪个类别就等于1, cp=1
                t[range(nb), tcls[i]] = cp
                # 计算真实标签t与预测标签ps的分类损失，FocalLoss
                lcls += BCEcls(ps[:, 5:], t)
        # 所有的框的objness损失，所匹配到的样本对应的objectness为giou，没有匹配到的为0
        lobj += BCEobj(pi[..., 4], tobj)

    # 各自的损失乘以权重
    lbox *= h["giou"]
    lobj *= h["obj"]
    lcls *= h["cls"]

    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}


def build_targets(p, targets, model):
    """
    根据输出的结果p和真实目标，将目标与anchors进行对应，返回对应后的分类分数、边界参数信息用于损失计算
        :param p: 模型的输出 [[batch_size, num_anchors, grid_x, grid_y, params], [....], [....]]
        :param targets: [image_idx, class, x, y, w, h]
        :param model:
        :return:
                tcls: 匹配到样本的类别
                tbox: 匹配到样本的坐标信息，包括anchors相对gt的坐标偏移量和targets的高度宽度
                anch： 匹配到的样本anchors的尺寸
                indices: 匹配到的图片索引、anchor的索引、grid_cell的索引
    """
    # num_targets，包含了这一个batch所有的targets
    nt = targets.shape[0]
    tcls, tbox, indices, anch = [], [], [], []
    #
    gain = torch.ones(6, device=targets.device).long()

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # p包含了3个feature map预测结果，也就是3个yolo layer, 分别为[89, 101, 113]
    for i, j in enumerate(model.yolo_layers):
        # 获取对应特征图上anchors，anchors_vect是已经缩放到对应特征图尺度上的大小
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        # 将p[i]中grid_x, grid_y, gain有[1, 1, 1, 1, 1, 1] -> [1, 1, 15, 15, 15, 15]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        # num anchors = 3
        na = anchors.shape[0]
        # 将anchors组成一个列表，与targets对应，也就是3*n_targets的列表，[[0, 1, 2], [0, 1, 2], [0, 1, 2], ....].T
        at = torch.arange(na).view(na, 1).repeat(1, nt)

        # 将anchors与targets进行匹配
        # targets * gain后，targets变成了绝对坐标
        a, t, offsets = [], targets * gain, 0
        if nt:
            # 计算anchor模板与所有targets的wh——iou来匹配样本，也就是将at那个表
            # wh_iou表示相对iou值，只使用各自的长和宽, 并不是按照anchor的位置计算，因为anchor只有3个(不考虑grid数量)
            # 计算得到的是一个[3, num_targets]的矩阵，满足条件的地方为True，不满足的为False
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp["iou_t"]
            # a是取出满足条件的anchor的所有，只由0,1,2组成
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6], 将j中对应true的位置提取出来，t就变成了anchor对应的target的信息了,
            # 这里anchors与target就对应了，t中存储的是targets，由于其包含了坐标，所以也就对应了grid_cell，a则对应的是第几号anchor(一共三个)
            a, t = at[j], t.repeat(na, 1, 1)[j]

        # targets前两个元素image_idx, class
        b, c = t[:, :2].long().T
        # 这两个元素表示x,y中心点坐标
        gxy = t[:, 2:4]
        # 这两个元素表示wh
        gwh = t[:, 4:6]
        # gxy取整就是grid_cell的左上角坐标，在yolov3中grid_cell的坐标表示左上角坐标，所以offset=0
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid_cell的索引
        # 将所匹配的信息都放入到indices中，image_idx, anchor_index, grid_coord(y, x)
        # clamp_防止越界
        indices.append((b, a, gj.clamp_(0, gain[3]-1), gi.clamp_(0, gain[2]-1)))
        # 坐标相对左上角坐标的偏移量，以及wh，这个就是真实值
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        # 匹配到的样本的anchors，也就是满足条件的anchors的大小
        anch.append(anchors[a])
        # 匹配到样本的class
        tcls.append(c)

        if c.shape[0]:
            assert c.max() < model.nc, "Model accept %g classes labeled from 0-%g, however you labelled a class %g. See" \
                                       "https:////github.com/ultralytics/yolov3/wiki/Train-Custom-Data" % (
                                        model.nc, model.nc - 1, c.max())

    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    预测信息包括了在img_size尺度上的坐标信息xywh，class score和confidence，
    进行非极大值抑制，同时将xywh转成了xyxy(还是img_size尺度)， 并且返回结果只有6个值，[xmin, ymin, xmax, ymax, obj, class]

    :param prediction: [batch_size, num_anchors, params]
    :param conf_thres:
    :param iou_thres:
    :param multi_label: 每一个box是否属于多个类别，False
    :param classes:
    :param agnostic:
    :param max_num:
    :return:
    """

    merge = False
    # 预测最大最小的限制
    min_wh, max_wh = 2, 4096
    time_limit = 10.0

    t = time.time()
    # num_classes
    nc = prediction[0].shape[1] - 5
    multi_label &= nc > 1

    output = [None] * prediction.shape[0]
    # 遍历batch内的每张image
    for xi, x in enumerate(prediction):
        # 取出objectness大于confidence的anchors
        x = x[x[:, 4] > conf_thres]
        # 再去掉小于wh和大于max_wh的anchors
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]
        # 如果去掉后没有目标框了
        if not x.shape[0]:
            continue

        # confidence = obj_conf * cls_conf
        x[..., 5:] *= x[..., 4:5]
        # box坐标转换
        box = xywh2xyxy(x[:, :4])
        # 针对每个类别进行NMS处理,将结果变成6维
        if multi_label:
            """
            test
            import torch
            # 假设有5个类别
            num_classes = 5
            # 创建一个预测结果张量x，假设有3个检测框
            x = torch.tensor([
                [1.2, 2.3, 3.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                [4.5, 5.6, 6.7, 0.2, 0.1, 0.9, 0.8, 0.7, 0.6, 0.5],
                [7.8, 8.9, 9.0, 0.4, 0.3, 0.2, 0.1, 0.95, 0.85, 0.7]
            ])
            # 设置置信度阈值
            conf_thres = 0.7
            # 找到置信度大于阈值的检测框的索引
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            new_x = torch.cat((x[i, :5], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
            """
            # 将输入的检测结果x中置信度高于conf_thres的检测框的索引（i，j）提取出来，i是框的索引(行索引),j是类别索引(类索引)
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            # 将第i个box的坐标、大于阈值的分数信息、所属类别class重新组合成x
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)

        else:
            # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # 经过阈值处理后如果没有框了
        n = x.shape[0]
        if not n:
            continue
        # 当agnostic为True时，表示进行类别无关的检测，即只关注目标的存在与否，而不考虑具体的类别
        c = x[:, 5] * 0 if agnostic else x[:, 5]
        # 将boxes缩放到很大的尺度，避免不同类别之间计算冲突，类似Faster RCNN中不同特征层上加上一个很大的offset
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]
        # 这里才是真正的使用极大值抑制处理
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        # 保留max_num个目标
        i = i[:max_num]

        if merge and (1 < n < 3E3):
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            except:
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break
    return output

