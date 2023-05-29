"""
该脚本完成3个功能：
1、根据训练集和验证集，生成相应的.txt文件，类似voc数据集中的train.txt, val.txt
2、创建data.data文件，主要保存num_classes, train.txt和val.txt的路径，以及labels.names文件的路径
3、根据yolov3-spp.cfg创建my_yolov3.cfg文件，修改其中的predictor filters和yolo classes的参数，因为这两个参数根据预测的类别数量决定的

"""

import os

train_annotation_dir = "./my_yolo_dataset/train/labels"
val_annotation_dir = "./my_yolo_dataset/val/labels"
classes_label = "./data/my_data_label.names"
cfg_path = './cfg/yolov3-spp.cfg'

assert os.path.exists(train_annotation_dir), "train_annotation_dir not exists..."
assert os.path.exists(val_annotation_dir), "cal_annotation_dir not exists..."
assert os.path.exists(classes_label), "classes_label not exists"
assert os.path.exists(cfg_path), "cfg_path not exists...."


def calculate_data_txt(txt_path, dataset_dir):
    # 拿到train和val,将其文件名保存在txt文件中
    with open(txt_path, "w") as w:
        for file_name in os.listdir(dataset_dir):
            if file_name == "classes.txt":
                continue

            img_path = os.path.join(dataset_dir.replace("labels", "images"),
                                    file_name.split(".")[0]) + ".jpg"
            line = img_path + "\n"
            assert os.path.exists(img_path), "file: {} not exists....".format(img_path)
            w.write(line)


def create_data_data(create_data_path, label_path, train_path, val_path, classes_info):
    # 创建data.data文件，记录train.txt、val.txt、num_classes、classes_dict.names的路径

    with open(create_data_path, "w") as w:
        w.write("classes={}".format(len(classes_info)) + '\n')
        w.write("train={}".format(train_path) + "\n")
        w.write("valid={}".format(val_path) + "\n")
        w.write("names=data/my_data_label.names" + "\n")


def change_and_create_cfg_file(classes_info, save_cfg_path="./cfg/my_yolov3.cfg"):
    """
    根据自己的数据集的类别，将yolov3结构改成所需要的结构，主要是分类层的参数，包括：
        1、预测的filters
        2、classes 参数
    :param classes_info:
    :param save_cfg_path:
    :return:
    """

    # 在原始文件中分类涉及到的，filters为[648, 742, 833]
    # 涉及到的classes信息为[658, 752, 843]
    filters_lines = [652, 746, 837]
    classes_lines = [662, 756, 847]
    cfg_lines = open(cfg_path, "r").readlines()

    for i in filters_lines:
        assert "filters" in cfg_lines[i-1], "filters param is not in line:{}".format(i-1)
        output_num = (5 + len(classes_info)) * 3
        cfg_lines[i-1] = "filters={}\n".format(output_num)

    for i in classes_lines:
        assert "classes" in cfg_lines[i-1], "classes param is not in line:{}".format(i-1)
        cfg_lines[i-1] = "classes={}\n".format(len(classes_info))

    with open(save_cfg_path, "w") as w:
        w.writelines(cfg_lines)


def main():
    # 将训练集和验证集生成相应的txt
    train_txt_path = "data/my_train_data.txt"
    val_txt_path = "data/my_val_data.txt"
    calculate_data_txt(train_txt_path, train_annotation_dir)
    calculate_data_txt(val_txt_path, val_annotation_dir)
    # 读取class_label.names文件，label放入列表中
    classes_info = [line.strip() for line in open(classes_label, "r").readlines() if len(line.strip()) > 0]
    # 创建data.data信息，将这些路径以及num_classes保存起来
    create_data_data("./data/data.data", classes_label, train_txt_path, val_txt_path, classes_info)

    # 根据所获取到的num_classes的信息，修改yolov3的结构，生成新的结构保存成为my_yolov3.cfg文件
    change_and_create_cfg_file(classes_info)


if __name__ == "__main__":
    main()
