import math
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def init_seed(seed=0):
    """
    在这个 init_seed 函数中，首先使用 torch.manual_seed 函数设置随机种子为给定的 seed 值。
    这将确保在随机数生成过程中使用相同的种子，从而使得结果可复现。接下来，检查 seed 是否等于0。
    如果 seed 等于0，意味着使用系统的随机种子，即随机种子会根据系统时间等进行初始化，每次运行
    都会生成不同的随机数。为了提高性能，cudnn.benchmark 被设置为 True，这将允许在运行时针
    对硬件自动选择最佳配置，从而加速深度学习模型的训练过程。同时，cudnn.deterministic
    被设置为 False，这将禁用一些可能影响性能的确定性计算，以便获得更高的计算速度。

    :param seed:
    :return:
    """

    torch.manual_seed(seed)

    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True


def time_synchronized():
    """
    该函数的主要作用是提供一个在计算机上获取时间戳的方法，并在支持CUDA的情况下确保时间同步，以便在需要测量时间间隔的地方使用。
    :return:
    """
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def model_info(model, verbose):

    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print("%5s %40s %9s %12s %20s %10s %10s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print("%5g %40s %9s %12g %20s %10.3g %10.3g" %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ", %1f GFLOPS" % (macs / 1e9 * 2)
    except:
        fs = ''

    print("Model Summary: %g layers, %g parameters, %g gradients%s"%(len(list(model.parameters())), n_p, n_g, fs))


