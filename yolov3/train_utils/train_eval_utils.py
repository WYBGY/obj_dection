import sys

from torch.cuda import amp
import torch.nn.functional as F

from build_utils.utils import *
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq, accumulate, img_size,
                    grid_min, grid_max, gs,
                    multi_scale=False, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    lr_scheduler = None
    # 当训练第一轮时，启用热身训练
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        #
        warmup_iters = min(1000, len(data_loader) - 1)
        # 热身训练的学习策略，只在epoch为0时指定，其它的学习策略在外部
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1

    mloss = torch.zeros(4).to(device)
    now_lr = 0.
    # num of batches
    nb = len(data_loader)

    for i, (imgs, targets, paths, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 统计从第0个epoch开始的所有batches
        ni = i + nb * epoch
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        # Multi scale, 在训练过程中随机调整尺寸
        if multi_scale:
            # 每训练accumulate个batch，就随机修改一次输入图片大小
            if ni % accumulate == 0:
                # 在给定的最大尺寸和最小尺寸之间随机找一个尺寸，并取32的整数倍。grid_min grid_max为(size_min // 32, size_max // 32)
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            # 缩放因子
            sf = img_size / max(imgs.shape[2:])
            if sf != 1:
                # 最大的边长需要进行缩放，要将长边和短边均调整为缩放后的尺寸
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                # 对图片进行调整
                imgs = F.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

        # 混合精度训练上下文管理，CPU中不起作用
        with amp.autocast(enabled=scaler is not None):
            pred = model(imgs)

            # loss
            loss_dict = compute_loss(pred, targets, model)
            losses = sum(loss for loss in loss_dict.values())
        # 对于多GPU要汇总多路输出
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 求和
        loss_reduced = sum(loss for loss in loss_dict_reduced.values())
        # 4个loss的值进行拼接
        loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                loss_dict_reduced["obj_loss"],
                                loss_dict_reduced["class_loss"],
                                loss_reduced)).detach()
        # 更新平均损失
        mloss = (mloss * i + loss_items) / (i + 1)

        if not torch.isfinite(loss_reduced):
            print("WARNING: non-finite loss, ending training", loss_dict_reduced)
            print("training image path: {}".format(",".join(paths)))
            sys.exit(1)

        # 对loss进行缩放，当不进行热身训练时accumulate = 64 / batch_size
        losses *= 1. / accumulate
        # backward， 先loss backward，然后optimizer的step，最后lr_scheduler的step()
        # 当采用amp时，特殊处理
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        # 每训练accumulate个batch更新一次optimizer，
        # 注：通常loss.backward()梯度计算之后就更新模型参数，或者通过loss.backward()的累加，来进行参数更新，但这里好像没有进行累加，只根据最后一个batch的梯度更新的参数。
        if ni % accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        # 这里的lr_scheduler只在热身训练时起作用，其他情况下为None，lr_scheduler在函数外执行
        if ni % accumulate == 0 and lr_scheduler is not None:
            lr_scheduler.step()
    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, coco=None, device=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    if coco is None:
        coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for imgs, targets, paths, shapes, img_index in metric_logger.log_every(data_loader, 100, header):
        imgs = imgs.to(device).float() / 255.0

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        # model 输出推理结果也就是经过yolo layer后的结果和原始模型输出结果
        pred = model(imgs)[0]
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)
        model_time = time.time() - model_time

        outputs = []
        # 逐张图片进行遍历
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                boxes = p[:, :4]
                # 将boxes还原回原图的尺寸，这样计算的mAP才准确
                boxes = scale_coords(imgs[index].shape[1:], boxes, shapes[index][0]).round()
            # 对于一张image而言，模型的预测结果，都已经处理完毕
            info = {"boxes": boxes.to(device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}

            outputs.append(info)
        # 一个batch的结果放入到res中
        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        evaluate_time = time.time()
        coco_evaluator.update(res)
        evaluate_time = time.time() - evaluate_time
        metric_logger.update(model_time=model_time, evaluate_time=evaluate_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats", metric_logger)
    coco_evaluator.synchronize_between_processes()

    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()

    return result_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_type = ["bbox"]
    return iou_type