import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import RoiHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork


class FasterRCNNBase(nn.Module):
    """
        Main class for Generalized R-CNN.

        Arguments:
            backbone: 主干网络
            rpn (nn.Module): rpn模块
            roi_heads (nn.Module): 接受rpn输出的proposal和backbone的features，预测类别和框的，也也就是FasterRCNN的尾端部分
            transform (nn.Module):
        """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
                Arguments:
                    images (list[Tensor]): 待处理图片
                    targets (list[Dict[Tensor]]): 每个图片的label，

                Returns:
                    result (list[BoxList] or dict[Tensor]): the output from the model.
                        During training, it returns a dict[Tensor] which contains the losses.
                        During testing, it returns list[BoxList] contains additional fields
                        like `scores`, `labels` and `mask` (for Mask R-CNN models).

                """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target['boxes']
                # 判断是不是tensor
                if isinstance(boxes, torch.Tensor):
                    # boxes为多个目标的框的Tensor[[xmin1, ymin1, xmax1, ymax1], [xmin2, ymin2, xmax2, ymax2], ....]
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be "
                                         "a tensor of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # 这行代码使用了 torch.jit.annotate() 函数，它的作用是给输入的变量显式地标注类型信息，以便在 JIT 编译过程中更好地进行优化
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        # 将每一张img的height和width存起来
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        # 对images和targets进行转换，后面再说
        images, targets = self.transform(images, targets)
        # images输入backbone，得到特征图
        features = self.backbone(images.tensors)
        # 将所得到的特征图层放进字典中，如果只有1层则编号为"0",若有多层则为"0","1","2","3",.....
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        # 将images，targets还有特征图输入到rpn网络中
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
        # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(images, features, targets)
        # 将特征图、建议框、图片尺寸(transform之后的尺寸)、labels输入到roi—head网络，也就是最终的分类和回归网络中
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # 网络的最后一部分，因为图片和预测框都是在resize的基础上进行的，因此要进行还原
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        """
        在下面这个代码段中，根据是否处于 Torch Script 模式下，使用不同的返回方式：

            如果处于 Torch Script 模式下，则返回损失（losses）和检测（detections）的元组；
            如果不处于 Torch Script 模式下，则调用 eager_outputs 方法进行返回。
        其中，eager_outputs 方法也是 Faster R-CNN 模型的一个方法，其作用是对损失和检测进行整合和包装，以便于输出和可读性。
        """
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


class TwoMLPHead(nn.Module):
    """
    这个类是roi——pool之后的两个全连接层的类
        in_channels: 输入的尺寸，faster rcnn中输入尺寸为7*7*out_channels
        representation_size (int): size of the intermediate representation
    """
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FasterRCNNPredictor(nn.Module):
    """
    这是box_head之后的一部分，一边用于分类，一边用于预测
    参数为：
        in_channels: 经两个全连接层后输出的channel个数
        num_classes: 包括背景在内的类别的数量
    """

    def __init__(self, in_channels, num_classes):
        super(FasterRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes*4)

    def forward(self, x):
        # x.shape(batch_size, 1024),要么(batch_size, 1024, 1, 1)
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]

        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
        Implements Faster R-CNN.
        模型的输入是images，Tensor形式，[batch_size, C, H, W], 图片是经过标准化的，范围0~1，每张图片的尺寸可以不一样。

        模型的训练和推理阶段有所不同
        在训练阶段，必须输入images和labels，其中1个image的labels包含[N, 4]大小的真实框，以及[N]个框所对应的label

        模型训练阶段返回一个字典，包括RPN网络的分类结果和回归结果，以及各自对应的损失

        而在推理阶段，模型只需要images输入，返回的是经过后处理（post-processed）的结果，结果是一个列表，列表中的每一个元素是1个image的结果，
        是字典形式的，这个字典包括[N，4]大小的预测框，[N]对应着每个框的预测的分类的label，以及[N]所预测label的分数

        Arguments:
            backbone (nn.Module): 主干网络，用于特征提取，返回特征图，所返回的特征图需包含out_channels属性，backbone所返回的feature map
                                  应该是一个有序字典形式（OrderedDict[Tensor]），表示特征图的层数。

            num_classes (int): 模型所预测的所有类别数量，包括背景类
            min_size (int): 图片rescale的最小尺寸
            max_size (int): 图片rescale的最大尺寸
            image_mean (Tuple[float, float, float]): 图片标准化的均值
            image_std (Tuple[float, float, float]): 图片标准化的方差

            rpn_anchor_generator (AnchorGenerator): rpn中，在特征图上产生anchors的模型
            rpn_head (nn.Module): rpn中用于分类和回归的网络部分
            rpn_pre_nms_top_n_train (int): 训练阶段，使用NMS之前所保留的proposals
            rpn_pre_nms_top_n_test (int): 测试阶段，在使用NMS之前所保留的proposals
            rpn_post_nms_top_n_train (int): 训练阶段，使用NMS后所保留的proposals
            rpn_post_nms_top_n_test (int): 测试阶段，使用NMS后所保留的proposals
            rpn_nms_thresh (float): 使用NMS所使用的阈值
            rpn_fg_iou_thresh (float): 在训练rpn时，proposal和groundtruth的Iou阈值，当大于该阈值时则被认为是正例（前景）
            rpn_bg_iou_thresh (float): 在训练rpn时，proposal和groundtruth的Iou阈值，当小于这个阈值时，则被认为为反例（背景）
            rpn_batch_size_per_image (int): 训练RPN时，所采样的anchors的batch_size
            rpn_positive_fraction (float): 训练RPN时，1个batch所包含的正例的比例
            rpn_score_thresh (float): score阈值。RPN在推理时，只有当score大于该阈值时的proposal才会返回

            box_roi_pool (MultiScaleRoIAlign): roi——pool模型，就是将feature map转成7*7大小的feature map
            the module which crops and resizes the feature maps in
                the locations indicated by the bounding boxes
            box_head (nn.Module): pool之后的特征抽取
            box_predictor (nn.Module): 抽取特征之后用于分类和回归

            box_score_thresh (float): score阈值。在最后当score大于这个阈值的proposal才会被输出
            box_nms_thresh (float): 用于推理阶段，NMS阈值
            box_detections_per_img (int): 一张图片输出的框的最大个数
            box_fg_iou_thresh (float): 在最后分类和回归阶段的Iou阈值，训练时，当大于该阈值才认为是正例（类别）

            box_bg_iou_thresh (float): 在最后分类和回归阶段的Iou阈值，训练时，当小于该阈值时才认为是反例
            box_batch_size_per_image (int): 在训练时，最后的分类阶段，所采样的proposals的数量（batch——size）
            box_positive_fraction (float): 训练时，最后的分类阶段，一个batch（proposals）所包含正例的比例
            bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
                bounding boxes
        """

    def __init__(self, backbone, num_classes=None,
                 min_size=800, max_size=1333, # 图片最小尺寸和最大尺寸
                 image_mean=None, image_std=None, # 图片归一化的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, # rpn锚点生成，是一个model
                 rpn_head=None, # rpn网络的头部，包括3*3卷积和分类、回归
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=2000, # 在NMS之前所保留的proposal数量
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=2000, # 在NMS之后所保留的proposal数量
                 rpn_nms_threshold=0.7, # 使用NMS处理时所使用的IOU阈值
                 rpn_fg_iou_threshold=0.7, rpn_bg_iou_threshold=0.3, # 采样时所使用的的正反例的阈值
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, # rpn训练时每张图片的选取的proposals的batch_size
                 rpn_score_threshold=0.0, # rpn输出proposal的score阈值，大于该阈值才输出
                 # Box Parameters
                 box_roi_pool=None, box_head=None, box_predictor=None, # 网络的后半部分，输入均为model
                 box_score_threshold=0.05, # 输出预测框的分数阈值，大于该阈值时才会输出
                 box_nms_threshold=0.5, # 在预测目标框的时候进行NMS的阈值
                 box_detections_per_img=100, # 每张图片输出目标框的最大数量，一般一张image的目标不会超过100个
                 box_fg_iou_threshold=0.5, box_bg_iou_threshold=0.5, # 采样正负样本的iou阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25, # 采样的batch—size数量和正样本比例
                 bbox_reg_weights=None):
        # backbone必须包含out_channels属性
        if not hasattr(backbone, "out_channels"):
            raise ValueError("backone should contain an attribute out_channels specifying the number of "
                             "output channels (assume to be the same for all the levels")
        # rpn_anchor_generator要么是一个实例化的AnchorGenerator对象，要么是None
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        # box_roi_pool要么是声明过的对象，要么是None
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        # 在回归阶段，num_classes必须为None，在分类阶段，num_classes不能为None
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        # 将保留框数量的参数放在字典里
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义RPN框架
        rpn = RegionProposalNetwork(rpn_anchor_generator, rpn_head,
                                    rpn_fg_iou_threshold, rpn_bg_iou_threshold,
                                    rpn_batch_size_per_image, rpn_positive_fraction,
                                    rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_threshold,
                                    score_threshold=rpn_score_threshold)

        # roi_pool
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', "1", "2", "3"],
                                              output_size=[7, 7],
                                              sampling_ratio=2)
        # roi_pool之后的两个全连接层
        if box_head is None:
            resolution = box_roi_pool.output_size[0] # 7
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)

        # box_head之后一边用于分类，一边用于预测
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FasterRCNNPredictor(representation_size, num_classes)

        # 上面是对每一块定义了一个实例化对象，结合在一起，输入到ROI—Head中，也就是网络的最后一整块
        roi_heads = RoiHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_threshold, box_bg_iou_threshold, # 0.5, 0.5
            box_batch_size_per_image, box_positive_fraction, # 512  0.25
            bbox_reg_weights,
            box_score_threshold, box_nms_threshold, box_detections_per_img # 0.05, 0.5, 100
        )
        # 归一化参数，没有的话就用经验
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        # 对images进行转换，打包成一个batch
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)




