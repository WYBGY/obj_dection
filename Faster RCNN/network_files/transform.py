import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional
from torch import nn, Tensor
import torchvision

from .image_list import ImageList


@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape)) # 获取这张image的最小尺度
    max_size = float(torch.max(im_shape)) # 获取这张image的最大尺度
    # 取最大的缩放比例，缩放后大小要均满足min—size和max—size
    scale_factor = self_min_size/min_size # 计算缩小比例

    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size/max_size
    # 使用torch.nn.function的interpolate进行插值resize
    image = torch.nn.functional.interpolate(image[None], scale_factor=scale_factor, mode="bilinear",
                                            recompute_scale_factor=True, align_corners=False)[0]
    return image


# 对box进行缩放，输入为image的原始尺寸和缩放后的尺寸
def resize_boxes(boxes, origin_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    # ratios为分别在h和w方向缩放的比例，比例不同
    ratios = [torch.tensor(s, dtype=torch.float32, device=boxes.device) /
              torch.tensor(s_origin, dtype=torch.float32, device=boxes.device)
              for s, s_origin in zip(new_size, origin_size)]

    ratios_height, ratios_width = ratios
    # unbind: 解绑操作,相当于xmin = boxes[:, 0], ymin = boxes[:, 1], ....
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    """
        转换图片和label的transform，包括：
        图片的标准化、resize以及label的resize
        返回一个ImageList以及targets的列表List[Dict[Tensor]]
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # mean本来是一维的，None表示复制一个维度
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息,这里是对一张image的处理
        Args:
            image: 输入的图片 [channel, height, width]
            target: 输入图片的相关信息（包括bboxes信息）[dict]

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        """
        torchvision._is_tracing()是一个私有函数，其作用是判断当前是否处于PyTorch的tracing模式。
        在这种模式下，会记录下模型的计算图以便后续进行优化或者导出为 TorchScript 代码。
        """
        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image, size, float(self.max_size))

        # 没有target时，返回resize的image
        if target is None:
            return image, target
        # 对boxes进行resize, [h, w]为原始尺寸，image是经过resize之后的
        bbox = target['boxes']
        bbox = resize_boxes(bbox, [h, w], image.shape[-2:])
        target['boxes'] = bbox
        return image, target

    # 辅助函数，从列表中找出每一个维度的最大值
    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将多张图片打包成一个batch，每个图片的尺寸可以是不同的，但是batch的每个tensor的shape是相同的
        :param images: 输入图片
        :param size_divisible: 将图像的高和宽调整到该数的整数倍
        :return: 打包成一个batch后的tensor数据
        """

        if torchvision._is_tracing():
            return self._onnx_batch_images(images, size_divisible)
        # 找出这一批图片中在每一个维度上的最大的尺寸[max_channel=3, max_height, max_width]
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # 将max_height向上调整至size_divisible的倍数，如max_size=63,则调整后为32*2=64
        max_size[1] = int(math.ceil(float(max_size[1])/stride) * stride)
        # max_width也是同样的操作
        max_size[2] = int(math.ceil(float(max_size[2])/stride) * stride)
        # batch的形状[batch_size, max_channel=3, max_height, max_width]
        batch_shape = [len(images)] + max_size

        # batch_imgs = torch.zeros(batch_shape)
        batched_imgs = images[0].new_full(batch_shape, 0)
        # 将图片放入batch中
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,   # type: List[Dict[str, Tensor]]
                    image_shapes, # type: List[Tuple[int, int]]
                    original_image_sizes # type: List[Tuple[int, int]]
                     ):
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果, len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:
            return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]['boxes'] = boxes

        return result

    # 开始处理
    def forward(self,
                images,      # type: List[Tensor]
                targets=None # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        # 逐个进行处理
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            # 图片标准化
            image = self.normalize(image)
            # 图片和target的resize
            image, target_index = self.resize(image, target_index)
            # 将转换后的再放回
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 处理完后，再打包成batch
        # resize之后的图片尺寸
        image_sizes = [img.shape[-2:] for img in images]
        # 将不同尺寸的image打包成一个batch
        images = self.batch_images(images)
        """
        torch.jit.annotate 是 PyTorch 提供的一种类型注解的方式，用于指定变量的类型和维度等信息，
        以便在 JIT 编译和优化的过程中提供更多的信息，从而提高性能和准确性。
        """
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)

        return image_list, targets