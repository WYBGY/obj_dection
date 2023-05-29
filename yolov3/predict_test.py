import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from build_utils.draw_box_utils import draw_objs


def main():
    # 推理时图输入尺寸，必须是32的整数倍
    img_size = 512
    cfg = "./cfg/my_yolov3.cfg"
    # 训练好的权重
    weight_path = "weights/yolov3spp-voc-512.pt"
    json_file = "./data/pascal_voc_classes.json"
    img_path = "test.png"

    assert os.path.exists(cfg), "cfg file {} does not exists.".format(cfg)
    assert os.path.exists(weight_path), "weights file {} does not exists.".format(weight_path)
    assert os.path.exists(json_file), "json file {} does not exists....".format(json_file)
    assert os.path.exists(img_path), "image file {} does not exists...".format(img_path)

    with open(json_file, "r") as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    weights_dict = torch.load(weight_path, map_location="cpu")
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict

    model.load_state_dict(weights_dict)
    model.to(device)

    model.eval()
    with torch.no_grad():
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)
        # BGR
        img_o = cv2.imread(img_path)
        assert img_o is not None, "Image not found" + img_path

        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # BGR转成RGB，然后再将channel放在前面 3*416*416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()
        # 预测结果包括了xywh(在new_shape尺度), confidence, class
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()
        print(t2 - t1)
        # 结果调用NMS对结果进行处理, 同时NMS将xywh转成了xyxy
        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print(t3 - t2)

        if pred is None:
            print("No target detected.")
            exit(0)

        # 处理检测结果,将在new_shape尺度上的框的坐标调整回原图尺度上
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        # 直接从array中读取Image
        pil_img = Image.fromarray(img_o[:, :, ::-1])
        plot_image = draw_objs(pil_img,
                               bboxes,
                               classes,
                               scores,
                               category_index=category_index,
                               box_thresh=0.2,
                               line_thickness=5,
                               font="arial.ttf",
                               font_size=20)
        plt.imshow(plot_image)
        plt.show()
        plot_image.save("test_result.jpg")


if __name__ == "__main__":
    main()
