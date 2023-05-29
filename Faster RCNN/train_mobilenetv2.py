import os
import datetime
import torch
import torchvision

import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
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
    device = torch.device("cpu")
    print("Using {} device training".format(device))

    # 用于保存result的文件
    result_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 保存权重的文件
    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = "./VOCtrainval_11-MAY-2012/"
    aspect_ratio_groups_factor = 3
    batch_size = 4
    amp = False # 是否使用混合精度训练，需GPU支持

    # 检查数据集
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise ValueError("VOCdevkit does not exist in path {}".format(VOC_root))
    # 数据集制作，包括图片读取、标签的xml解析
    train_dataset = My_datasets(VOC_root, "2012", data_transform["train"], "train.txt")
    train_sampler = None

    # 是否按图片相似高宽比采样图片，组成batch
    # 使用的话能够减小训练时所需的GPU显存，默认使用
    if aspect_ratio_groups_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间内的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_groups_factor)
        # 每个batch图片从同一高宽比例区间取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

    # num of workers
    nw = min([os.cpu_count(), batch_size if batch_size>1 else 0, 8])
    print("USing %g dataloader workers" % nw)

    if train_sampler:
        # 使用高宽采样图片时，dataloader需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # 验证集制作
    val_dataset = My_datasets(VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=21)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler() if amp else None

    train_loss = []
    learning_rate = []
    val_map = []
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # first frozen backbone and train 5 epochs                    #
    # 先冻结特征提取网络(backbone), 训练RPN以及最终预测网络部分           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    for param in model.backbone.parameters():
        param.requires_grad = False

    # 定义optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 先使用5个epoch训练RPN和最终预测的网络部分
    init_epochs = 5
    for epoch in range(init_epochs):

        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                          device, epoch, print_freq=50,
                                          warmup=True, scaler=scaler)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate,返回12个指标AP,AP50,AP75....
        coco_info = utils.evaluate(model, val_data_loader, device=device)

        # 将coco_info进行保存
        with open(result_file, 'a') as f:
            # 吧loss和lr也写进去
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, "   ".join(result_info))
            f.write(txt + '\n')

        val_map.append(coco_info[1])

    # 训练完5个epoch之后将结果存为预训练
    torch.save(model.state_dict(), "./save_weights/pretrain.pth")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # second unfrozen backbone and train all network                                      #
    # 解冻特征提取网络权重(backbone)， 训练整个网络                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # 冻结backbone底层的部分权重
    for name, parameter in model.backbone.named_parameters():
        split_name = name.split(".")[0]
        if split_name in ["0", "1", "2", "3"]:
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # 这个在前5个epoch中没有指定lr衰减策略，只在第0个epoch制定了热身训练衰减策略
    # 这里的lr_scheduler影响train_one_epoch的，这里进入到train_one_epoch后，lr_scheduler变成了None
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    num_epochs = 20
    for epoch in range(init_epochs, num_epochs + init_epochs, 1):

        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)

        train_loss.append(mean_loss)
        learning_rate.append(lr)
        # 在train_one_epoch中没有lr的step,其他的都更新了
        lr_scheduler.step()

        coco_info = utils.evaluate(model, val_data_loader, device=device)

        with open(result_file, 'a') as f:

            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()] + [f"{lr:.6f}"]]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + '\n')

        val_map.append(coco_info[1])

        # 保存最后5个epoch权重
        if epoch in range(num_epochs+init_epochs)[-5:]:
            save_files = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(save_files, "./save_weights/mobile-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    main()