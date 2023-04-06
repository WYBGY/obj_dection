"""
制作自己的数据集，这里以VOC数据集为例，自己的数据集可以在此基础上更改
VOC数据集是Images和XML组成的，XML是label文件

在分类中定义一个类比较简单，使用如下：
class my_data(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path     #拿取图片路径列表
        # self.label = label			#拿取标签列表
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):   #必须加载的方法
        img_after = Image.open(self.img_path[index]).convert('RGB')
        # label = self.label[index]
        if self.transform is not None:  #对图片进行二次处理
            img_after = self.transform(img_after)

        return img_after   #返回处理完的图片数据和标签

    def __len__(self):     #必须加载的方法,实际上好像没什么用
        return len(self.img_path)
"""


import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class My_datasets(Dataset):
    """
    root: 数据路径, './VOCtrainval_11-May-2012'
    year: VOC数据的年份， 2007和2012
    transformer：数据转换
    """
    def __init__(self, root, year='2012', transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", '2012'], "you must be in [2007, 2012]"
        #
        if "VOCdevkit" in root:
            self.root = os.path.join(root, f"VOC{year}")
        else:
            self.root = os.path.join(root, "VOCdevkit", f"VOC{year}")

        # 图片的路径
        self.img_root = os.path.join(self.root, "JPEGImages")
        # label的路径
        self.annotations_root = os.path.join(self.root, "Annotations")
        # 读取train.txt和val.txt
        txt_path = os.path.join(self.root, "ImagesSets", "Main", txt_name)
        assert os.path.exists(txt_path), f"not found {txt_name} file"
        # 将文件名对应到label的文件名，加xml
        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                        for line in read.readlines() if len(line.strip()) > 0]

        # 读取xml文件
        """
        这里是init,只存储了xml_path的列表，并没有对data进行存储，检查annotations中所包含的检测目标，将不符合object的删除掉
        """
        self.xml_list = []
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: '{xml_path}' not found, skip this annotations")
                continue
            with open(xml_path) as fid:
                xml_str = fid.read()
            # 解析xml格式的转成etree
            xml = etree.fromstring(xml_str)
            # 解析xml格式，变为字典，方法定义在后边
            data = self.parse_xml_to_dict(xml)['annotation']
            if "object" not in data:
                print(f"INFO: no object in '{xml_path}', skip this annotations")
                continue

            self.xml_list.append(xml_path)
        assert len(self.xml_list) > 0, "in {} file does not find any information".format(txt_path)

        # 读取类别信息
        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exists".format(json_file)

        with open(json_file, 'r')  as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # 这里要读取xml，并返回label的信息
        """
        data中的存储的image的label信息
        1个image中可能包含多个目标，因此data是一个list，每个list包含1个object的信息，包括box和label等

        """
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image:{} format not JPEG".format(img_path))

        # 处理label, 1个image含有多个box和label
        boxes = []
        labels = []
        iscrowd = []
        # 一个image有多个object
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            if xmax <= xmin or ymax <= ymin:
                print("Warning: in {} xml, there are some bbox w/h <= 0".format(xml_path))
                continue
            #
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if "difficult" in obj:
                iscrowd.append(int(obj['diffcult']))
            else:
                iscrowd.append(0)

        # 转成tensor格式
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        # 每个框的面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 将上面的所有的label信息打包成target
        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'iscrowd': iscrowd, 'area': area}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}
