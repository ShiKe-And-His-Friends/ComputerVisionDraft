'''
    2022-10-06 手写Baseline-YOLO模型

    致谢： https://github.com/bubbliiiing/yolov4-pytorch

'''
import os
import torch
from utils.dataloader import YoloDataset ,yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from torch.utils.data import DataLoader
from net.yolo import YoloBody

if __name__ == '__main__':
    # *********************************************************#
    # configure
    # *********************************************************#
    classes_path = 'E:/Torch/yolov4-pytorch-master/model_data/voc_classes.txt' # 分类数量
    anchors_path = 'E:/Torch/yolov4-pytorch-master/model_data/yolo_anchors.txt'
    train_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_train.txt' # 训练图片和路径
    val_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_val.txt'  # 验证图片和路径
    distributed = False # 指定是否单卡训练
    num_workers = 4 #多线程读取
    pretrained = False
    #是否进行冻结训练 #默认先冻结主干训练后解冻训练
    Freeze_Train = True
    Freeze_batch_size = 8
    unfreeze_batch_size = 16
    batch_size = Freeze_batch_size if Freeze_Train else unfreeze_batch_size
    shuffle = True if distributed else False
    input_shape = [416, 416]
    anchors_mask = [[6,7,8] ,[3,4,5] ,[0,1,2]] #用于帮助代码找到对应的先验框，一般不修改


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
    train_sampler = None # distributed False
    val_sampler = None

    train_dataset = YoloDataset(train_lines ,input_shape ,num_classes ,train = True)
    val_dataset = YoloDataset(val_lines ,input_shape ,num_classes ,train = False)
    gen = DataLoader(train_dataset ,shuffle= shuffle,batch_size= batch_size,num_workers= num_workers,pin_memory=True ,
                     drop_last=True ,collate_fn=yolo_dataset_collate ,sampler=train_sampler)
    val_gen = DataLoader(val_dataset, shuffle= shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

    # *********************************************************#
    ## yolo-conv2d-1
    # *********************************************************#
    model = YoloBody(anchors_mask ,num_classes ,pretrained)

    # *********************************************************#
    ##### epoch one
    # *********************************************************#
