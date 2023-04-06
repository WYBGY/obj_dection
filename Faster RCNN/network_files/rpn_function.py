from typing import List, Tuple, Dict, Optional, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[Any, Tensor]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorGenerator(nn.Module):
    """
    anchors生成器，用于在特征图上生成anchors的模块
    :arg
    sizes : Tuple[Tuple[int]]
    aspect_ratios: Tuple[Tuple[float]]
    模块可以计算在每个feature map在多个sizes和ratios下生成的anchors的数量

    注意，所输入的参数sizes和ratios必须长度一致，也就是说，sizes和ratios的每一维对应着在那一个feature map所产生的的anchor的参数
    比如，当有3个feature map的时候，len(sizes) = len(ratios)=3, 就是说每一个sizes对应着3个ratios
    sizes = ((128,), (256,), (512,))
    ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))
    或者
    sizes=((32, 64, 128, 256, 512),)
    ratios=((0.5, 1.0, 2.0),)


    sizes[i] 和 ratios[i] 也就是在第i个feature map生成anchor的参数可以是不相同的，
    在这个feature map 上每个cell可以生成sizes[i] * ratios[i]个anchors

    """
    __annotations__ ={
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()

        # 先做输入参数的格式判断，不符合进行转换, (128, 256, 512)转换成((128,), (256, ), (512, ))
        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s, ) for s in sizes)

        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios, ) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = None






