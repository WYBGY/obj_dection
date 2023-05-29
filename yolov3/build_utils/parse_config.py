"""
解析模型cfg文件，返回模型的字典形式
"""

import os
import numpy
import numpy as np


def parse_model_cfg(path: str):
    # 检查文件是否存在
    if not os.path.exists(path):
        raise ValueError("the cfg file not exists....")

    # 读取cfg文件
    with open(path, "r") as f:
        lines = f.read().split("\n")
    # 去掉空行和注释行
    lines = [x for x in lines if x and not x.startswith("#")]
    # 去掉首尾空格
    lines = [x.strip() for x in lines]

    mdefs = []
    for line in lines:
        # 判断是不是一个block，block是"[]"包裹的
        if line.startswith("["):
            # 如果是以"["开头，说明是一个新的block, 为这个block建立一个新的字典
            mdefs.append({})
            # 将block的名字存到"type"字段
            mdefs[-1]["type"] = line[1:-1].strip()
            # 将convolutional的batch_normalize重置为0, 后面再调整
            if mdefs[-1]["type"] == "convolutional":
                mdefs[-1]["batch_normalize"] = 0
        else:
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()

            if key == "anchors":
                # 对anchors进行特殊处理，去除中间的空格, 转成array
                # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                val = val.replace(" ", "")
                # 将anchors转成9*2的array
                mdefs[-1][key] = np.array([float(x) for x in val.split(",")]).reshape(-1, 2)
            elif (key in ["from", "layers", "mask"]) or (key == "size" and "," in val):
                # 对于shortcut, route, yolo层的参数不止1个转成整数list
                mdefs[-1][key] = [int(x) for x in val.split(",")]
            else:
                # TODO: .isnumeric() actually fails to get float case
                # 其他情况如果是数字，就转成整数或float，如果是字符就转成字符
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val
    # 检查所有key
    supported = ["type", "batch_normalize", "filters", "size", "stride", "pad", "activation", "layers", "groups",
               "from", "mask", "anchors", "classes", "num", "jitter", "ignore_thresh", "truth_thresh", "random",
               "strid_x", "stride_y", "weights_type", "weights_normalization", "scale_x_y", "beta_nms", "nms_kind",
               "iou_loss", "iou_normalizer", "cls_normalizer", "iou_thresh", "probability"]

    # 从1开始检查，0是net模块没有使用
    for x in mdefs[1:]:
        for k in x:
            if k not in supported:
                raise ValueError("Unsupported fields: {} in cfg".format(k))

    return mdefs


def parse_data_cfg(path):
    """
    解析data.data文件，拿出对应的内容
    :param path:
    :return:
    """

    if not os.path.exists(path) and os.path.exists("data" + os.sep + path):
        path = "data" + os.sep + path

    with open(path, "r") as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        key, val = line.split("=")
        options[key.strip()] = val.strip()

    return options
