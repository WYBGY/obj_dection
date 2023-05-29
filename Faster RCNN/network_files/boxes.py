import torch
from typing import Tuple
from torch import Tensor
import torchvision


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    将boxes进行裁减，主要是那些超出image边界的box，将其裁减回来
    因为每张image的顶点都是(0,0)，因此最大值就是height和width
    Arguments:
        :param boxes: [N, 4], (x1, y1, x2, y2)
        :param size: image shape [height, width]
    :return: [N, 4]
    """

    dim = boxes.dim()
    # x1, x2
    boxes_x = boxes[..., 0::2]
    # y1, y2
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(0, width)
        boxes_y = boxes_y.clamp(0, height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    移除很小的proposals，根据所给的最小尺寸，height和width都小于这个尺寸的框将会被滤除
    :param boxes: [N， 4]，[x1, y1, x2, y2]
    :param min_size: float
    :return: [N, 4]
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # height和width都大于min—size的才会被保留
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # 返回索引
    keep = torch.where(keep)[0]

    return keep


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
        Performs non-maximum suppression (NMS) on the boxes according
        to their intersection-over-union (IoU).

        NMS iteratively removes lower scoring boxes which have an
        IoU greater than iou_threshold with another (higher scoring)
        box.

        Parameters
        ----------
        boxes : Tensor[N, 4])
            boxes to perform NMS on. They
            are expected to be in (x1, y1, x2, y2) format
        scores : Tensor[N]
            scores for each one of the boxes
        iou_threshold : float
            discards all overlapping
            boxes with IoU > iou_threshold

        Returns
        -------
        keep : Tensor
            int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
        """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    非极大值抑制方法，从一系列重合度较高的框中选出最具有代表性的，当两个框的IOU超过threshold，认为是重叠的，保留最大的那个
    该方法中为了避免不同level所表示的框重叠的可能性，每个level引入一个偏移量

    :param boxes: [N, 4] proposals
    :param scores: [N] score
    :param idxs: [N]，每个proposal所属的level
    :param iou_threshold: IOU阈值，当超过这个阈值，则认为两个框是重叠的，保留score最大的那个
    :return: 返回经NMS之后保留的下来的proposal的索引
    """
    # 先看下boxes是不是空的，如果空的
    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)

    """# 为了保证在不同的level上出现proposals的重叠而导致IOU重叠大被误删除，在进行NMS时，需要在不同的level上单独进行
    # 在实现时，为每个boxes加上一个偏移量，使得在不同level上，拥有不同的坐标值，这样同时进行NMS时就不会出现不同level之间出现IOU过大的问题"""
    # 找出boxes中最大的坐标值 (x1, y1, x2, y2)中最大的值，是一个值
    max_coordinate = boxes.max()
    # 计算偏移量，相当于将每个level转换到level倍的坐标系
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上offsets
    boxes_for_nms = boxes + offsets[:, None]
    # 进行nms
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    计算两个box的iou值， iou = iner/(arae1 + area2 - iner)
    :param boxes1:
    :param boxes2:
    :return:
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou