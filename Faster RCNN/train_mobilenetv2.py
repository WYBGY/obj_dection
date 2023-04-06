import os
import datetime
import torch
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2, vgg
from my_datasets import My_datasets
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils


def create_model(num_classes):
    # 基础提取特征网络，采用预训练参数，只要提取后的特征层
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    # backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512
    backbone = MobileNetV2(weight_path="./backbone/mobilenet_v2.pth").features
    backbone.out_channels = 1280
    # 生成anchors，1个cell生成5*3个anchor
    # 注意输入均为tuple
    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1, 2.0),))
    # 这里是roi——pool层，输出7*7大小
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], # 要在哪个特征图上进行，mobile_net只有1个
                                                    output_size=[7, 7], # 输出大小， 7*7
                                                    sampling_ratio=2) # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def main():
    device = torch.device.