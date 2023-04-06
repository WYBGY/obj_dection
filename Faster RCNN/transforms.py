"""
这是一个自己实现的transforms的功能，其中Compose、ToTensor在torchvision的transfrom自带，这里实现了一遍
这里主要实现RandomHorizonFlip的功能。
Compose、ToTensor在分类中，使用示例如下：
dataset=my_data(image_path,   #注意my_data是我上面自己声明的一个类
                        transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((256, 256)),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                            )

"""

import random
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


# 这里对图片进行随机水平翻转，在识别中只需翻转图片，在检测中主要是box要跟着进行翻转
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target['boxes']
            # 对box进行翻转，水平翻转对于y轴(第1维ymin, 第3维ymax)没有影响
            # 用宽度减去原来的水平坐标，就是翻转后的坐标
            bbox[:, [0, 2]] = width - bbox[:, [0, 2]]
            target['boxes'] = bbox
        return image, target
