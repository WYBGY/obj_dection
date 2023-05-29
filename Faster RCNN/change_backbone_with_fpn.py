import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from my_datasets import My_datasets
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from backbone import BackboneWithFPN, LastLevelMaxPool


def create_model(num_classes):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor
    """
    create_feature_extractor是一种模型重构的方法，根据所提供的return layers,对backbone进行重构，
    跟BackboneWithFPN中IntermediateLayersGetter具有相似的功能，但是IntermediateLayersGetter方法只能获取第一层的子模块，
    在使用时create_feature_extractor更加灵活，仅限于1.10以上版本
    """
    # 直接使用自带的模型，作为backbone
    backbone = torchvision.models.mobilenet_v3_large(pretrained=True)

    # 找出所需要的layer，可以先打印backbone，然后找到对应的feature的name
    return_layers = {"features.6": "0",
                     "features.12": "1",
                     "features.16": "2"}

    # 每个特征层的output_channel
    in_channels_list = [40, 112, 960]
    # 使用create_feature_extractor进行backbone的重构
    new_backbone = create_feature_extractor(backbone, return_layers)

    # 重构完的backbone，输入到BackboneWithFPN模块中，最终构建出带有FPN的backbone，
    # regetter参数变为False，因为不需要再使用IntermediateLayersGetter进行重构

    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    # anchor_sizes设定，每个tuple的元素是在每一个feature map上的anchor尺度大小,具有多层fpn一般都是这样设定
    # 而在单层中(mobilenetV2)中一般为((64, 128, 256), ),因为对于FPN而言，越前面的层尺度越大，细节越多，更容易检测小目标
    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0), ) * len(anchor_sizes)

    anchor_generator = AnchorsGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2"],
                                                    output_size=[7,7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


# 更换backbone主要区别就是上面的那部分，其他的都一模一样
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training....".format(device))

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise ValueError("VOCdevkit does not in path {}".format(VOC_root))

    train_dataset = My_datasets(VOC_root, "2012", transforms=data_transform['train'], txt_name="train.txt")
    train_sampler = None

    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 所有image的长宽比的bins分布位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 创建batch，每个batch的image从同一个比例中取(还是同一个bins?)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])