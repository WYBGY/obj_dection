import datetime
import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import train_utils.train_eval_utils
from models import *
from build_utils.datasets import *
from build_utils.utils import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    # weights dir
    wdir = "weights" + os.sep
    best = wdir + "best.pt"
    # 用于保存训练结果的文件，包括12个coco评价指标
    result_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 已经转好的cfg路径
    cfg = opt.cfg
    # data的路径
    data = opt.data

    epochs = opt.epochs
    batch_size = opt.batch_size
    #
    accumulate = max(round(64 / batch_size), 1)
    # 预训练权重文件路径
    weights = opt.weights
    # 训练时会根据给定的img_size进行调整至67% ~ 150%
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size
    # 是否使用多尺度进行训练
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数
    gs = 32
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple"%(imgsz_test, gs)
    grid_min,  grid_max = imgsz_test//gs, imgsz_test//gs
    # 多尺度训练
    if multi_scale:
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667
        # 预测特征层大小(grid)也跟着相应的图片尺寸发生改变
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max
        print("Using multi_scale training, image range [{}, {}]".format(imgsz_min, imgsz_max))

    # 解析数据的路径，放在字典中
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    # num_classes
    nc = 1 if opt.single_cls else int(data_dict["classes"])
    # 修改超参数类别增益
    hyp["cls"] *= nc / 80
    hyp["obj"] *= imgsz_test / 320

    # remove previous results
    for f in glob.glob(result_file):
        os.remove(f)

    # 初始化模型
    model = Darknet(cfg).to(device)

    # 是否冻结权重，只训练predictor
    if opt.freeze_layers:
        # 当类型为yolo layer时，其前一层就是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list)
                                if isinstance(module, YOLOLayers)]

        # 冻结除了predictor和YOLOLayer之外的层
        freeze_layer_indices = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x-1 not in output_layer_indices)]

        for idx in freeze_layer_indices:
            for parameter in model.module_list[idx].parameters():
                # parameter.requires_grad = False 也可以
                parameter.requires_grad_(False)

    else:
        # 如果冻结参数为False，则默认训练Darknet53之后的部分
        # 如果训练全部权重，删除下面的代码
        darknet_end_layer = 74
        for idx in range(darknet_end_layer + 1):
            for parameter in model.module_list[idx].parameters():
                # parameter.requires_grad = False
                parameter.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    # 是否采用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    start_epoch = 0
    best_map = 0.0
    # 检查是否存在已经训练好的模型，如果有就加载训练好的
    if weights.endswith(".pt") or weights.endswith(".pth"):
        ckpt = torch.load(weights, map_location=device)

        try:
            """
            这行代码的作用是过滤和筛选模型权重字典中的项，仅保留那些在当前模型状态字典中具有相同元素数量的项。
            换句话说，这行代码的作用是删除那些模型权重字典中与当前模型状态字典中的元素数量不匹配的项，以确保
            权重字典与当前模型状态字典保持一致。这通常用于加载模型权重时，确保只加载与当前模型结构匹配的权重。
            """
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. "\
                "See https://github.com/ultralytics/yolov3/issues/657"%(opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # 读取存储的相关信息
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        if ckpt.get("training_results") is not None:
            with open(result_file, 'w') as file:
                file.write(ckpt["training_results"])

        # 如果start大于epochs，则再进行额外的epochs个fine tuning
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print("%s has been trained for %g epochs, Fine-tuning for %g additional epochs."%
                  (opt.weights, ckpt["epoch"], epochs))
            epochs += ckpt["epoch"]

        if opt.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        del ckpt
    # # 学习率下降策略，通过余弦退火的方法，是lr0在epochs内逐件降为lrf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # # 指定从哪个epoch开始
    scheduler.last_epoch = start_epoch
    #
    # # 来看一下lr的下降曲线, Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]["lr"])
    #
    # plt.plot(y, ".-", label="LambdaLR")
    # plt.xlabel("epoch")
    # plt.ylabel("LR")
    # plt.tight_layout()
    # plt.savefig("LR.png", dpi=300)

    # 加载数据集
    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        hyp=hyp,
                                        rect=opt.rect,
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls)

    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]) # number of workers

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # 除了rectangular启用时，shuffle均为True
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=val_dataset.collate_fn)
    # num classes注入进模型
    model.nc = nc
    # hyp注入到模型
    model.hyp = hyp
    # giou loss ratio
    model.gr = 1.0

    coco = get_coco_api_from_dataset(val_dataset)
    print("Starting training for %g epochs...." % epochs)
    print("Using %g dataloader workers" % nw)

    for epoch in range(start_epoch, epochs):
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少个batch 训练完64张图片
                                               img_size=imgsz_train,
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # imgsz_min // 32
                                               grid_max=grid_max,  # imgsz_max // 32
                                               gs=gs,
                                               print_freq=50,      # 训练多少个step打印一次
                                               warmup=True,
                                               scaler=scaler
                                               )

        scheduler.step()
        # 当需要进行评估或者最后一个batch时，进行evaluate
        if opt.notest is False or epoch == epochs - 1:
            #  coco = get_coco_api_from_dataset(val_dataset)
            result_info = train_util.evaluate(model, val_dataloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]
            voc_mAP = result_info[1]
            coco_mAR = result_info[8]

            # 将结果写到tensorboard, 4个loss, learning rate, mAP0.5, mAP0.5~0.95, mAR0.5~0.95
            if tb_writer:
                tags = ["train/giou_loss", "train/obj_loss", "train/cls_loss", "train/loss",
                        "learning_rate", "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # 将result info写进result files
            with open(result_file, "a") as f:
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            if coco_mAP > best_map:
                best_map = coco_mAP

            if opt.savebest is False:
                # 如果不要求保存最好的模型，则每个epoch都保存下来
                with open(result_file, "r") as f:
                    save_files = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "training_results": f.read(),
                        "epoch": epoch,
                        "best_map": best_map
                    }
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, './weights/yolov3spp-{}.pt'.format(epoch))
            else:
                #
                if best_map == coco_mAP:
                    with open(result_file, "r") as f:
                        save_files = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "training_results": f.read(),
                            "epoch": epoch,
                            "best_map": coco_mAP
                        }
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, best.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--cfg", type=str, default="cfg/my_yolov3.cfg", help="cfg_path")
    parser.add_argument("--data", type=str, default="data/data.data", help="*.data path")
    parser.add_argument("--hyp", type=str, default="cfg/hyp.yaml", help="hyperparameter path")
    parser.add_argument("--multi-scale", type=bool, default=True, help="adjust (67% - 150%) img_size every 10")
    parser.add_argument("--img_size", type=int, default=512, help="test_size")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--savebest", type=bool, default=False, help="only save best model")
    parser.add_argument("--notest", action="store_true", help="only test final epoch")
    parser.add_argument("--cache-images", action="store_true", help="cache images for faster training")
    parser.add_argument("--weights", type=str, default="weights/yolov3-spp-ultralytics-512.pt",
                        help="initial weights path")
    parser.add_argument("--name", default="", help="renames results.txt to results_name.txt if supplied")
    parser.add_argument("--device", default="cuda:0", help="device id (i.e. 0 or 0, 1 or cpu)")
    parser.add_argument("--single-cls", action="store_true", help="train as single-class datasets")
    parser.add_argument("--freeze-layers", type=bool, default=False, help="freeze non-output layers")
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print("Start Tensorboard with 'tensorboard --logdir=runs', view at http://localhost:6006/")
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)


