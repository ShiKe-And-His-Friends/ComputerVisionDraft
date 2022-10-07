'''
    2022-10-06 手写Baseline-YOLO模型

    致谢： https://github.com/bubbliiiing/yolov4-pytorch

'''
import os
import torch
from utils.dataloader import YoloDataset
from utils.utils import get_anchors, get_classes

if __name__ == '__main__':
    # *********************************************************#
    # configure
    # *********************************************************#
    classes_path = 'E:/Torch/yolov4-pytorch-master/model_data/voc_classes.txt' # 分类数量
    anchors_path = 'E:/Torch/yolov4-pytorch-master/model_data/yolo_anchors.txt'
    train_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_train.txt' # 训练图片和路径
    val_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_val.txt'  # 验证图片和路径

    input_shape = [416, 416]

    # *********************************************************#
    # dataloader
    # *********************************************************#

    # ----------------------------------------------------------#
    # 获取 classes 和 anchor 信息
    #----------------------------------------------------------#
    class_names , num_classes = get_classes(classes_path)
    anchors ,num_anchors = get_anchors(anchors_path)


    # ----------------------------------------------------------#
    # 构建训练数据 和 测试数据 的容器
    # ----------------------------------------------------------#
    with open(train_annotation_path ,encoding='utf-8') as f:
        train_lines = f.readlines()
    f.close()
    with open(val_annotation_path ,encoding='utf-8') as f:
        val_lines = f.readlines()
    f.close()
    num_train = len(train_lines)
    num_val = len(val_lines)

    train_dataset = YoloDataset(train_lines ,input_shape ,num_classes ,train = True)
    val_dataset = YoloDataset(val_lines ,input_shape ,num_classes ,train = False)

    # *********************************************************#
    ## yolo-conv2d-1
    # *********************************************************#

    # *********************************************************#
    ### cs-darknet-53
    # *********************************************************#

    # *********************************************************#
    ##### epoch one
    # *********************************************************#
