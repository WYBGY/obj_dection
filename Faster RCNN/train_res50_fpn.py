import os
import datetime

import torch

# 自定义的transforms
import transforms
from network_files import FasterRCNN, FasterRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_datasets import My_datasets
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils


def create_model(num_classes, load_pretrain_weight=True):
    """
    backbone默认使用FrozenBatchNorm2d, 目的防止batch_size过小导致的效果差
    trainable_layers ["layer4", "layer3", "layer2", "layer1", "conv1"]
    resnet50 imagenet weights url:https://download.pytorch.org/models/resnet50-0676ba61.pth
    """
    # 构建backbone
    backbone = resnet50_fpn_backbone(pretrain_path='./backbone/resnet50.pth',
                                     norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己的数据时不修改这里的91，修改的是传入的num_classes
    model = FasterRCNN(backbone, num_classes=91)

    if load_pretrain_weight:
        # 载入预训练模型
        #  https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)

        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys:", missing_keys)
            print("unexpected_keys:", unexpected_keys)

    # 获取搭建的模型用于分类和回归的特征的channel，也就是分类前的特征图的channel
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 将分类的那部分替换为自己数据集的类别
    model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training".format(device))

    # 用于保存训练过程中的结果的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise ValueError("VOCdevkit does not in path {}.".format(VOC_root))

    train_dataset = My_datasets(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    # 是否按照图片的长宽比进行采样组成batch
    # 如果使用可以节省显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图片的长宽比，在bins区间中的索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每一个bacth从长宽比bins中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print("Using %g dataloader workers" % nw)

    if train_sampler:
        # 当采用按照长宽比取batch的时候, 有batch_sampler参数
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory = True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    val_dataset = My_datasets(VOC_root, "2012", data_transform['val'], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    # 创建模型
    model = create_model(num_classes=args.num_classes)
    model.to(device)

    # 定义optimizer,先定义需要训练的参数，然后定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                               lr=args.lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 定义学习率下降策略
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.3)

    # 开始训练前，查看上次是否保存了权重，如果指定了地址，则读取，接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        print("the training process from epoch{}.....".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):

        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        lr_scheduler.step()
        # 返回15个指标，map、mar等
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        with open(results_file, "a") as f:
            result_info = {f"{i:.4f}" for i in coco_info + [mean_loss.item()] + [f"{lr:.6f}"]}
            txt = "epoch: {} {}".format(epoch, "   ".join(result_info))
            f.write(txt + '\n')

        val_map.append(coco_info[1])

        # 保存模型
        save_files = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        }

        if args.amp:
            save_files['scaler'] = scaler.state_dict()

        torch.save(save_files, "./save_weights/resNetFPN-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot map curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    # device类型
    parser.add_argument("--device", default="cpu", help="device")
    # VOC训练数据集
    parser.add_argument("--data-path", default="./VOCtrainval_11-May-2012/", help="dataset")
    # num_classes
    parser.add_argument("--num-classes", default=20, type=int, help="num_classes")
    # save_dir
    parser.add_argument("--output-dir", default="./save_weights", help="path where to save")
    # 上次训练保存的文件地址
    parser.add_argument("--resume", default="", type=str, help="resume from checkpoint")
    # start epoch
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")
    # epochs
    parser.add_argument("--epochs", default=15, type=int, metavar="N", help="number of total epochs to run")
    # lr
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    # momentum
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    # weight_decay
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay"
                        , dest="weight_decay")
    # batch_size
    parser.add_argument("--batch-size", default=8, type=int, metavar="N", help="batch size")
    # aspect-ratio-group-factor
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    # amp
    parser.add_argument("--amp", default=False, help="use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)





