from typing import Optional, List, Tuple, Dict

import torch
from torch import Tensor
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    计算faster rcnn的损失，大致和RPN中的损失一样，不同的是，在fastrcnn中，最后的类别不再是只有背景和前景，而是包含每个类别
    就是说class_logits和labels的维度为[N, 21], 其中每一维表示属于某一个类别的分数/标签
    Arguments：
        :param class_logits: 预测类别信息，[num_proposals, num_classes]
        :param box_regression: 预测的4个回归参数， [num_proposals, 21*4]
        :param labels: proposals所匹配到的gt的真实标签, [batc_size, num_proposals_per_image, 1]
        :param regression_targets: proposals与所匹配到的gt，计算得到的回归参数[batch_size, num_proposals_per_image, 4]
        :return:
    """
    # 将所有image的labels拼接在一起
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    # 计算分类损失，labels没有经过one_hot
    classification_loss = F.cross_entropy(class_logits, labels)

    # 接下来计算回归损失
    # 因为一开始regression——targets是匹配到所有正负样本，且存在一部分gt=0的那部分,使用labels信息，只拿出正样本
    # labels中大于0的样本为正样本
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]
    # 正样本的类别
    labels_pos = labels[sampled_pos_inds_subset]
    # num_proposals, num_classes
    N, num_classes = class_logits.shape
    # box_regression是为每个类别都有预测一个回归参数，这里重新reshape一下
    box_regression = box_regression.reshape(N, -1, 4)
    # 计算回归损失，输入为正样本所属类别的预测回归参数和真实的回归参数
    # 因为box_regression包含了正负样本，且为每个类别都预测了回归参数，这里需要取出正样本及其多对应类别的回归参数
    box_loss = det_utils.smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos],
                                        regression_targets[sampled_pos_inds_subset],
                                        beta=1/9,
                                        size_average=False) / labels.numel()
    return classification_loss, box_loss


class RoiHeads(torch.nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_match": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler
    }

    def __init__(self,
                 box_roi_pool,  # roi_pool部分，生成7*7大小的feature map
                 box_head,  # TwoML Head部分，两个全连接模块
                 box_predictor,  # 分类和回归部分模块，
                 # 用于Faster RCNN训练部分的参数
                 fg_iou_threshold, bg_iou_threshold,  # 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # 每张image的采样数量，正样本数量
                 bbox_reg_weights,  # 回归权重
                 # 在Faster RCNN推理时的参数
                 score_threshold,  # 分数阈值
                 nms_threshold,  # NMS阈值
                 detection_per_image  # 每张图片检测最大数量
                 ):
        super(RoiHeads, self).__init__()
        # 计算iou的方法实例化
        self.box_similarity = box_ops.box_iou
        # 匹配anchors和gt的方法
        self.proposal_matcher = det_utils.Matcher(fg_iou_threshold, bg_iou_threshold, allow_low_quality_matches=False)
        # 正负样本采样方法
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        # 编码和解码的方法，用于回归参数计算和框的计算
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
        # roi_pool模块
        self.box_roi_pool = box_roi_pool
        # 两个全连接模块
        self.box_head = box_head
        # 分类和回归模块
        self.box_predictor = box_predictor
        # 参数定义
        self.score_threshold = score_threshold
        self.detection_per_image = detection_per_image
        self.nms_threshold = nms_threshold

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(['labels' in t for t in targets])

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gtboxes拼接到proposals的后面，目的是为了在训练时正样本不足或者忽略ground truth的情况

        """
        proposals = [torch.cat((proposal, gt_box))
                     for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        将proposals与gt按照iou进行匹配，返回每个proposals对应的gtbox的索引和类别信息
        假设1个image的proposals为[10,4](xmin, ymin, xmax, ymax), gt为[2, 4], 那么返回regression_targets[10, 1]
        每一维对应着proposals的回归参数， labels[10, 21], 每一维对应着proposals对应的类别信息（21类）
        :param proposals:
        :param gt_boxes:
        :param gt_labels:
        :return:
        """
        matches_idxs = []
        labels = []
        # 遍历每个image
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            # 图片中没有标注框, 则这张图proposals对应的gt为0，类别信息为0
            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )

            else:
                # 计算proposals与gt_boxes的iou矩阵
                matched_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                # 根据iou的阈值，进行匹配，这里阈值low和high均为0.5
                # 匹配后返回proposals对应的gt的索引, below 的为-1， between为-2
                matches_idxs_in_image = self.proposal_matcher(matched_quality_matrix)
                # 限制最小值,这里将-1，-2调整到0，这样在计算类别时，防止出现越界
                # 在计算每个proposal类别时，这些等于0的会对应的gt索引为0的类别上，后续处理
                clamped_matched_idxs_in_image = matches_idxs_in_image.clamp(min=0)
                # proposals对应的gt的label信息,
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # 背景类，即iou小于low threshold的proposals
                bg_inds = matches_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                # 将背景类别置为0，这里同时会对上面的-1、-2置为0的那一部分进行调整
                labels_in_image[bg_inds] = 0
                # 忽略的样本索引,将其置为-1
                ignore_inds = matches_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLD
                labels_in_image[ignore_inds] = -1
            # 这里实际上有一部分设置为-1和-2的都对应着gt索引为0的gt，不用担心，labels中存储了这一部分信息，labels中>0的才是正样本
            matches_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matches_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        """
        样本采集，利用fg_bg_sampler方法采集正负样本的proposals的索引，然后再拼接在一起
        :param labels:
        :return:
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将索引进行拼接，应该可以使用和RPN中一样的方法，这里采用遍历的方法实现的，没有将batch的所有image进行拼接
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)

        return sampled_inds

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        """
        在训练时需要计算proposals与targets的回归参数用于损失计算，需要划分正负样本和对应的类别信息
        :param proposals: RPN网络所得到的回归参数，将anchors调整成为的proposals
        :param targets: labels信息
        :return: 返回proposals的分类结果和回归参数，以及损失
        """
        # 检查targets是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device
        # 获取gt的信息
        gt_boxes = [t['boxes'].to(dtype) for t in targets]
        gt_labels = [t['labels'] for t in targets]

        # 将ground truth添加到proposals中,防止在训练初期采集不到正样本
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # 然后将proposals与gt_box进行匹配，返回匹配后的每个proposals对应的gt_box和分类信息，和RPN中的一样，只是这里的类别不再是背景和前景
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # 然后正负样本采集，从labels中进行采集，并将采集到的样本拼接在一起
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)
        # 遍历每张图片，取出所采样的样本的类别，并得到所对应的proposals对应的gt的坐标
        #########################################################################################
        # 这里其实可以仿照RPN中进行，直接在assign_targets_to_proposals的方法，直接返回匹配后的坐标信息，    #
        # 然后再从sampled的索引取出对应的labels信息和坐标信息即可，但这里可能为了减少计算量，没有计算每一个     #
        # proposals, 但这个方法中所返回的regression_targets包含了负样本的回归参数                       #
        #########################################################################################
        for img_id in range(num_images):
            # 每张图片的正负样本的索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本所对应的proposals的坐标,
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应的labels信息
            labels[img_id] = labels[img_id][img_sampled_inds]

            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            # 当gtbox为空时，这时proposals匹配到的gt的坐标全为0
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 返回proposals所匹配到的gt_box的坐标，这里包含了正样本和负样本，不太合理
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        # 回归参数计算了负样本的回归参数，在计算loss时会处理，但显然不太合理
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,     # type: Tensor
                               box_regression,   # type: Tensor
                               proposals,        # type: List[Tensor]
                               image_shapes      # type: List[Tuple[int, int]]
                               ):
        """
        预测的class_logits、box_regressin进行处理，生成boxes[xmin, ymin, xmax, ymax]、labels、score
        具体步骤如下：
            （1） 根据proposals和box_regression, 将proposals调整成boxes；
            （2） 对预测的结果class_logits进行softmax处理；
            （3） 裁减boxes，将越界的box裁减回边界上
            （4） 移除所有背景信息
            （5） 移除低概率目标
            （6） 移除小尺寸目标
            （7） 执行NMS，并按照scores进行排序
            （8） 根据scores排序返回前top-n个目标
        Arguments:
            :param class_logits: 预测分类信息，每一个proposal所属21个类别的“scores”[num_proposals, 21]
            :param box_regression: 预测proposals的回归参数，每个proposals对于每个类别都有一个回归参数，[num_proposals, 21*4]
            :param proposals: proposals坐标[xmin, ymin, xmax, ymax]，shape[batch_size, num_proposals_per_image, 4]
            :param image_shapes: 图片经过resize之前的尺寸
            :return:
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        # 每张image的proposals的数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据box_regression和proposals计算boxes
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测的class_logits进行softmax处理
        pred_scores = F.softmax(class_logits, -1)
        # 将boxes和scores按照batch_size进行划分，一一对应
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        # 下面一段类似RPN中的proposals的返回
        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张image
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 对预测到的boxes进行裁减
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # 对每个预测的proposals划定类别
            # labels为[0~21]
            labels = torch.arange(num_classes, device=device)
            # 然后将labels扩展成为与scores shape一致的，scores为[num_proposals, 21], 那么labels就是[0~21] * 21
            labels = labels.view(1, -1).expand_as(scores)

            # 移除背景类，背景类的class_label = 0
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # 原先每一行表示一个proposals所对应的信息，num_proposals * 20
            # 现在将这些信息进行展平处理，也就是变成了20*num_proposals[......|.......|.......|........|]
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 滤除低score的目标
            inds = torch.where(torch.gt(scores, self.score_threshold))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 执行NMS处理, 这里之前的levels隔离变成了labels隔离
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_threshold)
            keep = keep[:self.detection_per_image]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        return all_boxes, all_scores, all_labels

    def forward(self,
                features,  # type: Dict[str, Tensor]
                proposals,  # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None,  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Faster RCNN的后一般全部内容，整合了roi——pool、2*fc、predictor，
        将backbone的feature maps和RPN产生的proposals经过这里，输出最终预测结果和损失
        Arguments:
            :param features: 经过backbone所产生的feature maps，OrderedDict格式
            :param proposals: 经过RPN所生成的proposals，也就是RPN产生的回归参数，将anchors调整后的东西
            :param image_shapes: 图像的尺寸信息，高和宽，resize之后的
            :param targets: label信息
            :return: 返回结果和损失
        """

        # 先检查targets
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t['boxes'].dtype in floating_point_types, "targets boxes must of float type"
                assert t['labels'].dtype == torch.int64, "targets label must of int64 type"

        # 训练时要进行样本的划分，计算边界回归参数(真实值)和标签
        if self.training:
            # 正负样本采样，保留采样后的proposals，和每个proposals对应的labels信息以及回归参数信息(回归参数包含了正负样本)
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            # 推理时则无需使用labels和regression
            labels = None
            regression_targets = None

        # 将采集的proposals 和feature map送入到roi_pool层，得到每个box的feature，7*7[num_proposals, channel, 7, 7]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # 将box特征输入到全连接网络 [num_proposals, representation_size]
        box_features = self.box_head(box_features)
        # 然后输入到分类和回归的网络中
        class_logits, box_regression = self.box_predictor(box_features)

        # 接下来计算损失、返回结果
        # result包含着每个proposals的分类信息、分数、回归参数
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # 计算分类损失和回归损失
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            # 在推理阶段不需要计算损失，需要返回box、labels和score信息，这些boxes要经过处理
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "scores": scores[i],
                        "labels": labels[i]
                    }
                )

        return result, losses



