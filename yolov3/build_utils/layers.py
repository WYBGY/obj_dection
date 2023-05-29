import torch
import torch.nn.functional as F
from .utils import *


class FeatureConcat(nn.Module):
    """
    将多个特征矩阵在channel维度上进行concatenate拼接

    """
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers
        self.multiple = len(layers) > 1

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):
    """
    将特征对应的位置进行相加add
    """
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers
        self.weight = weight
        # 所要融合的特征图的数量
        self.n = len(layers) + 1
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)
            x = x * w[0]

        nx = x.shape[1]
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i+1] if self.weight else outputs[self.layers[i]]
            na = a.shape[1]

            if nx == na:
                x = x + a
            elif nx > na:
                x[:, :na] = x[:, :na] + a
            else:
                x = x + a[:, nx]
        return x