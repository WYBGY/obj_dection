import argparse
import parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov5.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default='', help="model.yaml path")
    # parser.add_argument("--data", type=str, default='data/coco128.yaml')
    parser.add_argument("--data", type=str, default="data/data.data", help="data paths file")
    parser.add_argument("--hyp", type=str, default="cfg/hyp.yaml", help="hyperparameter path")
    parser.add_argument("--epochs", type=int, default=300, help="total training epoch")
    parser.add_argument("--batch_size", type=int, default=16, help="total batch_size ")
    parser.add_argument("--imgsz", '--img', '--img-size', type=int, default=640, help="train, val image size")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    """
    指定了--resume选项的解析方式。
    nargs="?"--resume选项可以接受零个或一个参数值。
    const=True表示如果--resume选项没有提供参数值，则使用默认值True。
    default=False表示如果--resume选项没有出现，则使用默认值False。
    """
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add


    parser.add_argument("--multi-scale", type=bool, default=True, help="adjust")
