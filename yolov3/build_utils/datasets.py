import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from build_utils.utils import xyxy2xywh, xywh2xyxy


help_url = "https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data"
img_formats = ['.bpm', '.jpg', '.jpeg', '.png', '.tif', '.dng']

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取image的原始尺寸
    通过exif的orientation信息判断图像是否旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始的size
    """
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        # 没有旋转信息
        pass
    return s


class LoadImagesAndLabels(Dataset):
    """
    yolo数据处理流程，繁琐且复杂，包括
        1. 读取文件列表txt文件。
        2. 初始化参数，包括：样本数量、num_batch、img_size、augument、hyp、rect、mosaic等
        3、查看shapes文件，没有的话进行读取和存储
        4、判断是否开启rect训练，如果开启，则计算宽高比，重新分配batch，每一个batch具有差不多的宽高比，并将每个batch的shape变为32的倍数
        5、cache labels，第一次逐个读取label文件，判断丢失、空、有效、重复等label文件，然后当文件数量大于1000时，存储成npy文件，
                        下次直接加载进来；
        6、判断是否从label中提取bbox，如果需要进行处理后，返回bbox(xmin, ymin, xmax, ymax)
        7、是否缓存images到内存中，逐一读取图片，放入self.imgs中，并保存对应的hw0和缩放到img_size后的h和w

    """
    def __init__(self,
                 path,  # 数据的路径，指向train.txt的路径
                 img_size=416,  # 预处理后输出的图片尺寸,test时设置的是最终使用的网络大小,在训练时设定的是多尺度,最大尺寸
                 batch_size=16,
                 augment=False, # 图像的增强，训练时为True，验证时为False
                 hyp=None, # 超参数
                 rect=False, # 是否使用矩形训练rectangular training
                 cache_images=False, # 是否缓存图片到内存中
                 single_cls=False, pad=0.0, rank=-1
                 ):
        try:
            # 读取训练文件列表
            path = str(Path(path))
            if os.path.isfile(path):
                with open(path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("%s does not exists"%path)

            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
            self.img_files.sort()
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        n = len(self.img_files)
        assert n > 0, "No images found in %s, see %s"%(path, help_url)

        # 将数据划分成一个个batch中去, 记录每个样本所属batch
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        # 总的batch数目
        nb = bi[-1] + 1
        self.n = n  # 样本总数量
        self.batch = bi  # 图片所属batch的记录
        self.img_size = img_size  # 预处理后输出的尺寸
        self.augment = augment  # 是否使用图像增强
        self.hyp = hyp
        self.rect = rect  # 是否使用rectangular training
        # 注意，开启rect时，mosaic默认关闭的
        self.mosaic = self.augment and not self.rect
        # 图片的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]

        # 查看data目录下是否存有.shapes文件，保存了每张image的width、height
        sp = path.replace(".txt", ".shapes")
        try:
            with open(sp, "r") as f:
                s = [x.split() for x in f.read().splitlines()]
                # 判断shapes文件中的数量是否与images的数量相等，不相等则重新生成
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:
            # 只在主进程(多GPU)显示进度，其他的rank img——files不经过tqdm包装
            if rank in [-1, 0]:
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                image_files = self.img_files

            s = [exif_size(Image.open(f)) for f in image_files]
            # 保存所有images的shape信息为.shapes文件
            np.savetxt(sp, s, fmt="%g")

        # 记录每张图片的原始尺寸
        self.shapes = np.array(s, dtype=np.float64)

        # Rectangular training https://github.com/ultralytics/yolov3/issues/232
        # 如果为True，训练网络时，会使用类似原图像比例的矩形
        # 开启rect后，mosaic默认关闭
        if self.rect:
            s = self.shapes
            # 计算高宽比
            ar = s[:, 1] / s[:, 0]
            # 将高宽比按照大小排序，返回索引值
            # 排序后的每个batch的高宽比大小就差不多
            irect = ar.argsort()
            # 根据排序后的高宽比，将图片、标签和shape进行重新排序
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            # 每个batch采用统一尺寸
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                # 第i个batch的长宽比最大值最小值
                mini, maxi = ari.min(), ari.max()
                # 如果高宽比小于1(w > h)，将w设置为img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高宽比大于1(w < h), 将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1/mini]
            # 设置成32的整数倍
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        self.imgs = [None] * n
        # label: [class, x, y, w, h]
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n

        extract_bounding_boxes, labels_loaded = False, False
        nm, nf, ne, nd = 0, 0, 0, 0 # num_missing, num_found, num_empty, num_duplicate
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"

        # 如果之前保存过npy文件，直接读取
        if os.path.isfile(np_labels_path):
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        # 遍历label文件
        for i, file in enumerate(pbar):
            if labels_loaded is True:
                # 如果存在缓存直接从缓存中读取
                l = self.labels[i]
            else:
                # 从label文件读取标签信息
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    # 当读取标签文件失败时
                    print("An error occured while loading the file {}: {}".format(file, e))
                    nm += 1 # file missing
                    continue

            # 读取到标签后再进行判断和检查
            if l.shape[0]:
                # 读取到标签的信息必须是5个值
                assert l.shape[1] == 5, "> 5 label columns: %s"%file
                assert (l >= 0).all(), "negative labels: %s"%file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s"%file

                # 检查每一行是否标签重复
                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    nd += 1
                # 单一类别模式，则把样本的class都改为0,这里不涉及
                if single_cls:
                    l[:, 0] = 0

                #
                self.labels[i] = l
                nf += 1   # file found
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    # 遍历image中每个框
                    for j, x in enumerate(l):
                        f = "%s%sclassifier%s%g_%g_%s"%(p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)

                        # 将相对坐标转为绝对坐标
                        # b: x, y, w, h
                        b = x[1:] * [w, h, w, h]
                        # 将宽高设置为二者最大值
                        b[2:] = b[2:].max()
                        # 放大裁减目标的宽高
                        b[2:] = b[2:] * 1.3 + 30
                        # 将坐标格式从x，y,w,h → xmin, ymin, xmax, ymax
                        b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                        # 裁减bbox坐标到图片内
                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)

                        assert cv2.imwrite(f, img[b[1]: b[3], b[0]: b[2]]), "Failure extracting classifier boxes"
            # 如果l为空的，没有目标
            else:
                ne += 1

            if rank in [-1, 0]:
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)"%(
                            nf, nm, ne, nd, n)

        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        # 当labels没有保存，并且样本数量大于1000，就把labels保存成npy格式
        if not labels_loaded and n > 1000:
            print("Saving labels to %s for faster future loading"%np_labels_path)
            np.save(np_labels_path, self.labels)

        # 将图片存入缓存，加速训练，但也会导致占用内存过大
        if cache_images:
            # 记录占用内存
            gb = 0
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = self.load_image(i)
                gb += self.imgs[i].nbytes
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb/1E9)

        # 检测损坏图片
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except Exception as e:
                    print("Corrupted image detected: {}, {}".format(file, e))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        hyp = self.hyp
        if self.mosaic:
            #
            img, labels = self.load_mosaic(index)
            shapes = None
        else:
            img, (h0, w0), (h, w) = self.load_image(index)
            # 启用rect时,找到对应的shape，不启用时shape则为img_size,训练时走上面
            shape = self.batch_shapes[self.shapes[index]] if self.rect else self.img_size
            # 将图片缩放到指定尺寸
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            # for COCO mAP rescaling
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            labels = []
            x = self.labels[index]
            if x.size > 0:
                # 对标签进行修正，与load mosaic类似,转成xmin, ymin, xmax, ymax
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # 如果只进行增强，就随机仿射变换，因为mosaic中已经进行了仿射变换跳过
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])
            # 无论进不进行mosaic，都hsv增强
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        nL = len(labels) # num_labels
        if nL:
            # 将xyxy转到xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # 标准化0—1
            labels[:, [2, 4]] /= img.shape[0] # height
            labels[:, [1, 3]] /= img.shape[1] # width

        if self.augment:
            # 再进行一次随机水平和垂直翻转
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                # 标签跟着翻转
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
            # 垂直翻转，一般不进行
            up_filp = False
            if up_filp and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
        # 标签本来是5个值，现在变成6个
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # 将BGR转成RGB， HWC转成CHW
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """
        该方法是专门为cocotools统计标签准备，不对图像和标签做任何处理
        :param index:
        :return:
        """
        # wh -> hw
        o_shapes = self.shapes[index][::-1]
        x = self.labels[index]
        labels = x.copy() #
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes, index = zip(*batch)
        # 之前将标签变为6个值，这里将第一个值变成索引
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index


    def load_image(self, index):
        img = self.imgs[index]
        if img is None:
            path = self.img_files[index]
            img = cv2.imread(path)
            assert img is not None, "Image Not Found" + path
            # origin wh
            h0, w0 = img.shape[:2]
            r = self.img_size/max(h0, w0)
            if r != 1:
                interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return img, (h0, w0), img.shape[:2]
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]


    def load_mosaic(self, index):
        """
        将四张图像拼接在一张马赛克图像中
        :param index: 图像的索引
        :return:
        """
        # 拼接后的labels
        labels4 = []
        s = self.img_size
        # 随机初始化拼接图像的中心点坐标，因为拼接后期望拼接后的大小与img_size一致,因此先填进去再裁减
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        # 从dataset中随机抽取3张图像进行拼接,图像的索引
        indices = [index] + [random.randint(0, len(self.labels)-1) for _ in range(3)]
        # 遍历4张图像进行拼接
        for i, index in enumerate(indices):
            # 读取图片
            img, _, (h, w) = self.load_image(index)

            #
            if i == 0:  # top left
                # 创建一张空图像，尺寸为s*2
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                # 中心点对着第一张图片的右下角，找出左上角坐标x1a, y1a
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # 上面对到左上角裁减后，再对应到原图中的坐标，因为要把原图中对应的位置截出来，放到左上角
                # 原图中，右下角坐标就是w, h，左上角为实际宽(高)与截取剩余宽度(高度)的差值
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h

            elif i == 1: # top right
                # 同理右上图像，左下角坐标对应着xc，yc
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: # bottom left
                # 同理对于左下角的图像，xc, yc对应着图像的在右上角
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a)

            else: # bottom right
                # 右下角图像，xc，yc对应着左上角
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                #
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 将提取到的位置都放到对应的位置中，a对应b
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # 计算填充后与边界的空隙，可能有的图片小，填充后还有空余,如果为负，说明填满了越界
            padw = x1a - x1b
            padh = y1a - y1b

            #
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:
                # 将labels信息转换成绝对坐标信息，原来label是原图中的相对坐标
                #
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)

        if len(labels4):
            # 标签进行拼接
            labels4 = np.concatenate(labels4, 0)
            # 防止标签越界
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        img4, labels4 = random_affine(img4, labels4,
                                      degrees=self.hyp["degress"],
                                      translate=self.hyp["translate"],
                                      scale=self.hyp["scale"],
                                      shear=self.hyp["shear"],
                                      border=-s//2)

        return img4, labels4


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考我写的博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，img4.shape/2
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and scale
    # 生成旋转及缩放矩阵
    R = np.eye(3)
    # 随机翻转角度
    a = random.uniform(-degrees, degrees)
    # 随机缩放因子
    s = random.uniform(1-scale, 1+scale)#
    # 旋转和缩放矩阵
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # 生成平移矩阵
    T = np.eye(3)
    # x translate
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border
    # y translate
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border

    # 生成错切矩阵
    S = np.eye(3)
    # x shear
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    # y shear
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # 将转换矩阵合起来
    M = S @ T @ R
    if (border != 0) or (M != np.eye(3)).any():
        # 进行仿射变换
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # 图像变换后对相应的label也要进行变换
    n = len(targets)
    if n:
        # 4个坐标，3个变换维度
        xy = np.ones((n * 4, 3))
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        # [4 * n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # 对transform后的bbox进行修正，假设变换后变成菱形，修正成矩形
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        # 翻转后的框也跟着翻转了，变回成正的框
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # 对坐标进行裁减，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的bbox面积
        area = w * h
        # 计算调整前bbox的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]
    return img, targets


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:
    :param new_shape:
    :param color:
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 新旧图片缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1]/shape[1])
    # only down 对于大于指定大小的图片进行缩放，小于的不变
    if not scale_up:
        r = min(r, 1.0)

    # 计算padding
    ratio = r, r
    # 新的缩放后的尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # 指定图片尺寸与缩放后的尺寸的wh 差值
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        # 取余保证padding之后是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    # 直接将图片缩放到指定尺寸
    elif scale_fill:
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    # 将padding分为上下、左右两部分
    dw /= 2
    dh /= 2
    # shape: [h, w] new_unpad: [w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 上下左右padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    # 这里可以参考我写的博文:https://blog.csdn.net/qq_37541097/article/details/119478023
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    # h,s,v色域的值,注意是BGR
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # uint8
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.uint16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
