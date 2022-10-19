'''
    2022-10-06 手写Baseline-YOLO模型

    致谢： https://github.com/bubbliiiing/yolov4-pytorch

'''
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import YoloDataset ,yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from torch.utils.data import DataLoader
from net.yolo import YoloBody
from net.yolo_training import YoloLoss
from net.yolo_training import get_lr_scheduler

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
    pretrained = False #  是否对主干Backbone进行训练，不训练则直接加载model_path
    #是否进行冻结训练 #默认先冻结主干训练后解冻训练
    Freeze_Train = True
    Freeze_batch_size = 8
    Unfreeze_batch_size = 16
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    shuffle = True if distributed else False
    Init_lr = 1e-2 #模型学习率
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-7
    save_period = 10 #多少次epoch保存一次权值
    input_shape = [416, 416]
    anchors_mask = [[6,7,8] ,[3,4,5] ,[0,1,2]] #用于帮助代码找到对应的先验框，一般不修改

    model_path = 'E:/Torch/yolov4-pytorch-master/model_data/yolo4_weights.pth' # 训练好的权值路径，SOTA数据结果
    save_dir = 'logs' # 保存权值和日志文件
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model = YoloBody(anchors_mask ,num_classes ,pretrained=pretrained)
    if not pretrained:
        print("wight_init()")
    if not model_path == '':
        print("LOAD weigth file {}".format(model_path))
        # 根据预训练的全职key和weight进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path ,map_location=device)
        load_key ,no_load_key ,temp_dict = [] , [] ,{}
        v_item = 0
        for k ,v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k])== np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else :
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        #TODO not match keys

    yolo_loss = YoloLoss(anchors ,num_classes ,input_shape
        ,False #cude
        ,anchors_mask
        ,0.005 #focal_loss
        ,0.25 #focal_alpha
        ,2 #focal_gamme
        ,'ciou' #iou_type
    )
    time_str = datetime.datetime.strftime(datetime.datetime.now() ,'%Y_%m_%d_$H_%M_%S')
    log_dir = os.path.join(save_dir ,"loss_" + str(time_str))
    #loss_history = LossHistory() #TODO save weights and logs
    loss_history = None

    # torch1.2 不支持amp torch1.7以上支持fp16
    scaler = None

    model_train = model.train()

    #TODO 多卡bn 和或者cuda

    # *********************************************************#
    #  读取数据集对应的txt
    # *********************************************************#
    with open(train_annotation_path ,encoding = 'utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path ,encoding ='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    #TODO 总训练世代

    # *********************************************************#
    #  主干特征提取网络特征通用，冻结训练可以加速训练速度
    #  也可以防止训练初期权值被破坏
    #  Init_Epoch为起始世代
    #  Freeze_Epoch为冻结训练的世代
    #  UnFreeze_Epoch为总训练世代
    #  提示OOM或者显存不足需调小Batch_size
    # *********************************************************#
    if True:
        UnFreeze_flag = False
        # 冻结一部分训练
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        # 样本集大小
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 判断样本集合大小，自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type in ['adam' ,'adamw'] else 5e-2
        lr_limit_min = 3e-4 if optimizer_type in ['adam' ,'adamw'] else 5e-4
        Init_lr_fit = min(max(batch_size / nbs* Init_lr ,lr_limit_min) ,lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr ,lr_limit_min * 1e-2) ,lr_limit_max * 1e-2)

        # 根据optimizer_type 选择优化器
        pg0 ,pg1 ,pg2 = [] ,[] ,[]
        for k,v in model.named_modules():
            if hasattr(v ,"bias") and isinstance(v.bias ,nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v,nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v ,"weight") and isinstance(v.weight ,nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam' : optim.Adam(pg0 ,Init_lr_fit ,betas =(momentum ,0.999)),
            'adamw': optim.AdamW(pg0 ,Init_lr_fit ,betas = (momentum ,0.999)),
            'sgd': optim.SGD(pg0 ,Init_lr_fit ,momentum = momentum ,nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params":pg1 ,"weight_decay":weight_decay})
        optimizer.add_param_group({"params":pg2})

        # 学习率下降的公式
        lr_scheduler_func = get_lr_scheduler()

        for epoch in range(Init_Epoch ,UnFreeze_Epoch):
            fit_one_epoch(model_train ,model .yolo_loss ,loss_history ,eval_callback ,optimizer ,epoch ,epoch_step ,epoch_step_val ,gen ,gen_val ,Unfreeze_epoch ,Cuda ,fp16 scaler ,save_period ,save_dir ,local_rank)

        if local_rank == 0:
            loss_history.writer.close()