import math
import sys
import time

import torch

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):

    model.train()
    """
        一个MetricLogger类，用于记录模型训练过程中的指标，包括损失、精度等
    """
    metric_logger = utils.MetricLogger(delimiter="   ")
    """
    SmoothedValue这个类的作用是跟踪一系列数值，并提供对这些数值进行平滑处理后的结果。它可以计算在一个窗口内的平均数、中位数、最大值等
    等统计信息，并且还可以计算整个序列的平均数。这个类适用于需要实时记录并平滑处理一系列数值的应用场景，比如训练神经网络时需要记录损失值
    或准确率等指标的变化情况。
    """
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    # 当训练第一轮时，启用热身训练
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0/1000
        # 热身训练的轮次
        warmup_iters = min(1000, len(data_loader)-1)
        """
        warmup_lr_scheduler这个函数的作用是为给定的优化器(optimizer)添加一个warmup策略的学习率调度器。
        在神经网络训练中，warmup通常是指在训练的开始阶段，将学习率逐步升高，以达到更好的训练效果。
        这个函数的实现就是根据当前的迭代数，返回一个学习率倍率因子，用于更新当前的学习率。当迭代次数小于给定的
        warmup_iters时，学习率倍率因子逐步增加，当迭代次数大于等于warmup_iters时，学习率倍率因子为1，即不
        再进行warmup。
        """
        # 启用热身训练的学习率调整策略
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)
    """
        1、当前 epoch 中已经处理的 batch 数量和总的 batch 数量，使用进度条的形式显示进度
        2、估计完成整个 epoch 的剩余时间（ETA，estimated time of arrival），格式为 hh:mm:ss
        3、每个 meter 的值，使用逗号分隔，例如 "loss: 0.1234, accuracy: 0.5678"
        4、迭代时间，即处理一个 batch 的平均时间，单位为秒
        5、数据加载时间，即读取一个 batch 的平均时间，单位为秒
        6、（可选）当前 GPU 显存占用情况，单位为 MB
    """
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images和targets放到列表中
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 混合精度训练上下文管理器，CPU环境中不起作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        """
        这个函数是用来在分布式训练中进行梯度聚合的。在多个GPU上训练模型时，每个GPU都计算一部分梯度，并将它们传递给主进程进行聚合。
        reduce_dict函数就是在主进程中聚合各个进程计算的梯度，保证每个进程都拥有同步的梯度。
        该函数输入一个字典，其中包含了需要聚合的梯度。如果是单GPU训练，那么直接返回原字典；
        如果是多GPU训练，那么先将字典中的值拼成一个张量，然后通过all_reduce函数对它们进行求和或求平均（取决于average参数），
        并将结果存储在一个新字典中，最后返回新字典。这个新字典的键与原字典相同，但它的值已经被聚合了。
        """
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = loss_reduced.item()
        # 记录训练损失,更新平均损失
        mloss = (mloss*i + loss_value) / (i+1)

        # 当损失为inf时停止训练
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        #
        optimizer.zero_grad()
        # 使用混合精度时的参数更新
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        # 不使用混合精度时的参数更新
        else:
            losses.backward()
            optimizer.step()

        # 在热身训练阶段，lr_scheduler不为None,当正常训练是lr_scheduler为None，由外部主导
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="   ")
    header = "Test"
    """
    这个功能可以帮助我们在评估模型性能时使用COCO评价指标（如mAP），因为COCO评价指标需要用到COCO API对象。
    get_coco_api_from_dataset(dataset)函数则判断输入的数据集dataset是不是torchvision.datasets.CocoDetection类型的，
    如果是则直接返回dataset.coco，否则调用convert_to_coco_api函数将其转换为COCO API对象。
    包含了 COCO API 中计算的 12 种指标的结果：
        AP (Average Precision): 平均精度。
        AP50 (Average Precision with IoU threshold 0.5): 在 IoU 阈值为 0.5 时的平均精度。
        AP75 (Average Precision with IoU threshold 0.75): 在 IoU 阈值为 0.75 时的平均精度。
        APs (Average Precision with small objects): 对小目标的平均精度。
        APm (Average Precision with medium objects): 对中等大小目标的平均精度。
        APl (Average Precision with large objects): 对大目标的平均精度。
        AR (Average Recall): 平均召回率。
        AR1 (Average Recall with 1 detection per image): 每张图像只允许一次检测的平均召回率。
        AR10 (Average Recall with 10 detections per image): 每张图像只允许 10 次检测的平均召回率。
        AR100 (Average Recall with 100 detections per image): 每张图像只允许 100 次检测的平均召回率。
        ARs (Average Recall with small objects): 对小目标的平均召回率。
        ARm (Average Recall with medium objects): 对中等大小目标的平均召回率。
        ARl (Average Recall with large objects): 对大目标的平均召回率。
    
    """
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 使用cpu进行评估时，跳过gpu相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)
        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        # 更新评估器结果
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # 多GPU下 metric_logger的汇总
    metric_logger.synchronize_between_processes()
    print("Average stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # 将每张图片的结果汇总，然后累加到当前实例的总结果中
    # 注意是，实例是多个，对应不同的IOU阈值，都有一个evaluator实例
    coco_evaluator.accumulate()
    # 输出评估结果的总结信息，包括各种IoU指标的精度、召回率、F1分数等
    coco_evaluator.summarize()

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()
    return coco_info


def _get_iou_types(model):
    """
    这段代码定义了一个函数 _get_iou_types，用于获取模型的 IoU 类型。如果模型是分布式数据并行模型
    （torch.nn.parallel.DistributedDataParallel），则会获取模型的 module；否则获取模型本身。
    最终返回一个字符串列表，其中只包含 "bbox"，表示使用的是边界框 IoU。
    :param model:
    :return:
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    iou_types = ['bbox']
    return iou_types
