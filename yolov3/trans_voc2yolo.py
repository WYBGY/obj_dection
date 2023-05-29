"""
将VOC数据的xml格式转换成yolo的txt格式，并将图像文件复制到相应文件夹
将标签文件的json格式转换成.name文件

"""

import os
from tqdm import tqdm
from lxml import etree
import json
import shutil


voc_root = "./VOCtrainval_11-May-2012/VOCdevkit"
voc_version = "VOC2012"

# 转换时用到的训练样本和验证样本，样本的名称保存在txt文件中
train_txt = "train.txt"
val_txt = "val.txt"

# 转换完成后，将新的数据集保存目录
save_file_root = "./my_yolo_dataset"

# VOC的label的json文件
label_json_path = './data/pascal_voc_classes.json'

# VOC数据集的路径，images、labels、txt目录
voc_images_path = os.path.join(voc_root, voc_version, "JPEGImages")
voc_xml_path = os.path.join(voc_root, voc_version, "Annotations")
train_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", train_txt)
val_txt_path = os.path.join(voc_root, voc_version, "ImageSets", "Main", val_txt)

# 检查文件是否存在
assert os.path.exists(voc_images_path), "VOC images path not exists...."
assert os.path.exists(voc_xml_path), "VOC xml path not exists...."
assert os.path.exists(train_txt_path), "VOC train.txt path not exists...."
assert os.path.exists(val_txt_path), "VOC val.txt path not exists...."

if os.path.exists(save_file_root) is False:
    os.makedirs(save_file_root)


def parse_xml_to_dict(xml):
    """
    VOC数据集的annotations为xml文件，将xml文件解析，转成字典的形式，参考tensorflow的recursive_parse_xml_to_dict
    这个函数和Faster RCNN中的一致，都需要解析etree
    :param xml:
    :return:
    """

    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != "object":
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 可能有多个目标，需要将目标放入列表中
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val="train"):
    """
    这个函数就是将xml文件转成yolo使用的txt文件

    :param file_names:
    :param save_root:
    :param class_dict:
    :param train_val:
    :return:
    """
    # 标签保存路径 "./my_yolo_dataset/train/labels"
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    # 图片保存路径 "./my_yolo_dataset/train/images"
    save_images_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} file....".format(train_val)):
        # 检查图片是否存在
        img_path = os.path.join(voc_images_path, file + ".jpg")
        assert os.path.exists(img_path), "file: {} is not exists...".format(img_path)
        # 检查xml是否存在
        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file: {} is not exists....".format(xml_path)
        # 读取xml
        with open(xml_path) as fid:
            xml_str = fid.read()

        xml = etree.fromstring(xml_str)
        # 解析xml
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data['size']["height"])
        img_width = int(data['size']["width"])

        # 将解析的数据写进txt
        assert "object" in data.keys(), "file: {} lack of object key.".format(xml_path)
        if len(data['object']) == 0:
            # 如果object没有目标
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
            continue

        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data['object']):
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                # 类别在
                class_index = class_dict[class_name] - 1

                # 进一步检查数据，标注数据存在w、h为0的情况，导致数据计算时回归损失变为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning in '{}' xml, there are some bbox w/h <= 0".format(xml_path))
                    continue

                # 将bbox信息转成yolo格式, yolo格式保存的相对中心点坐标和高度、宽度
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin
                # 绝对坐标转成相对坐标
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        path_copy_to = os.path.join(save_images_path, img_path.split(os.sep)[-1])
        if os.path.exists(path_copy_to) is False:
            shutil.copyfile(img_path, path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open("./data/my_data_label.names", "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # 读取json类别文件
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # 读取train.txt文件，训练文件名放入列表
    with open(train_txt_path, "r") as r:
        train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]

    # 将xml转成yolo的txt
    translate_info(train_file_names, save_file_root, class_dict, "train")

    # val.txt的文件如法炮制
    with open(val_txt_path, "r") as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]

    translate_info(val_file_names, save_file_root, class_dict, "val")

    # 创建.names文件
    create_class_names(class_dict)


if __name__ == "__main__":
    main()
