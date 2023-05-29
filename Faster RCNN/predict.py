import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pylab as plt

# 系统自带的transform
from torchvision import transforms
from network_files import FasterRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs


def create_model(num_classes):

    # 创建backbone
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone, num_classes, rpn_score_threshold=0.5, box_score_threshold=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))

    # 创建模型
    model = create_model(num_classes=21)

    weights_path = './save_weights/model.pth'
    assert os.path.exists(weights_path), "{} file does not exist".format(weights_path)

    weights_dict = torch.load(weights_path, map_location="cpu")
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict

    model.load_state_dict(weights_dict)
    model.to(device)

    label_json_path = "./pascal_voc_classes.json"
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    original_img = Image.open('./test3.jpg')

    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)

    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        print(model(init_img))

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("Interference time: {}".format(t_end - t_start))

        predict_boxes = predictions['boxes'].to("cpu").numpy()
        predict_classes = predictions['labels'].to("cpu").numpy()
        predict_scores = predictions['scores'].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=5,
                             font="arial.ttf",
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()


if __name__ == "__main__":
    main()


