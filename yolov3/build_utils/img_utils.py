"""
在dataset函数中实现过，将图片缩放到指定尺寸

"""

import numpy as np
import cv2


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放到指定尺寸大小
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

    # 缩放比例
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    # 如果不进行放大操作，只进行缩小，则取ratio 和 1的最小值
    if not scale_up:
        r = min(r, 1)
    # height和width缩放比例都为r
    ratio = r, r
    # 原图乘以缩放比例后的尺寸，new_unpad [h, w]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # 需要填充的dw、dh, 可以为正也可以为负，unpad第0维是height，与new_shape相反
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    #  如果启用rectangular，保持原图的高宽比例不变
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    # 之间简单粗暴的调整到指定大小
    elif scale_fill:
        # 不进行填充，直接缩放到指定尺寸
        dw, dh = 0, 0
        # new_unpad [w, h]
        new_unpad = new_shape[::-1]  # h, w -> w, h
        ratio = new_shape[1] / shape[1], new_shape[0]/shape[0]

    # 将需要填充的pad分为上下两部分
    dw /= 2
    dh /= 2
    # shape [h, w], new_unpad [w, h], 原图与缩放后的不一致，就进行resize
    # 因在cv中，[w, h]形式，所以要进行颠倒
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)




