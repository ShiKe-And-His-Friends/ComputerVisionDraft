'''
    2022-10-06 手写Baseline-YOLO模型

    致谢： https://github.com/bubbliiiing/yolov4-pytorch

'''
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from utils.dataloader import YoloDataset ,yolo_dataset_collate
from utils.utils import get_anchors, get_classes ,show_config
from torch.utils.data import DataLoader
from net.yolo import YoloBody
from net.yolo_training import YoloLoss
from net.yolo_training import get_lr_scheduler
from net.yolo_training import weight_init
from net.yolo_training import set_optimizer_lr
from utils.callback import LossHistory ,EvalCallback
from utils.utils_fit import fit_one_epoch

if __name__ == '__main__':
    #TODO revisit this configure variable values

    # *********************************************************#
    # configure
    # *********************************************************#
    classes_path = 'E:/Torch/yolov4-pytorch-master/model_data/voc_classes.txt'
    anchors_path = 'E:/Torch/yolov4-pytorch-master/model_data/yolo_anchors.txt'

    train_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_train.txt' # 训练图片和路径
    val_annotation_path = 'E:/Torch/yolov4-pytorch-master/2007_val.txt'  # 验证图片和路径
    Cuda = True # 是否使用GPU
    num_workers = 4 #多线程读取
    # ------------------------------------------------------------------------------------------------------------------------------------#
    #  训练分两个部分，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的设备的训练需求。
    #  冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch 等于 UnFreeze_Epoch ，此时仅进行冻结训练
    #
    #  提供若干参数设置建议，训练时根据需求灵活调整
    #   （一） 整个模型的预训练权重开始训练
    #       Adam：
    #           Init_Epoch = 0 , Freeze_Epoch = 50 , UnFreeze_Epoch = 100 , Freeze_Train = True ,opeimizer = 'adam' ,Init_lr = 1e-3 ,weight_decay = 0 (冻结)
    #           Init_Epoch = 0 , UnFreeze_Epoch = 100 ,Freeze_Train = False ,optimizer_type = 'adam' ,Init_lr = 1e-3 ,weight_decay = 0 （不冻结）
    #       SGD:
    #           Init_Epoch = 0 , Freeze_Epoch = 50 ,UnFreeze_Epoch = 300 , Freeze_Train = True ,optimizer_type = 'sgd' ,Init_lr = 1e-2 ,weight_decay = 5e-4 (冻结)
    #           Init_Epoch = 0 ,UnFreeze_Epoch = 300 ,Freeze_Train = False ,optimizer_type = 'sgd' ,Init_lr = 1e-2 ,weight_decay=5e-4 (不冻结)
    #   （二） 从主干网络的预训练权重开始训练
    #       Adam SGD 参数同上
    #       其中：由于从主干网络预训练权重开始训练，主干的权值不一定适合目标检测，需要更多的训练跳出局部最优解
    #           UnFreeze_Epoch 可以设置150-300之间调整 ，YOLOv5与YOLOX均推荐300
    #           Adam相对于SGD收敛快一些，因此UnFreeze_Epoch理论上可以小一点，但是推荐更多的Epoch
    #   （三） 从0开始训练：
    #       Init_Epoch = 0 ,UnFreeze_Epoch >= 300 ,Unfreeze_batch_size >=16 ,Freeze_Train = False （不冻结训练）
    #       其中: UnFreeze_Epoch 尽量不小于300。 optimizer_type = 'sgd' ,Init_lr = 1e-2 ,mosic = True
    #   (四) batch_size 的设置
    #       在显卡接受的范围内，以大为好。现存不足与数据集大小无关吧。提示显存不足（OOM或CUDA out of memory）只能调小batch_size
    #       受到BatchNorm层影响，batch_size 最小为2 ，不能为1
    #       正常情况下Freeze_batch_size 建议为Unfreeze_batch_size 的1-2倍，不建议设置差距过大，因为关系到学习率的自动调整
    #
    # ------------------------------------------------------------------------------------------------------------------------------------#
    # -------------------------------------------------------#
    #   冻结阶段训参数
    #       此时模型的主干被冻结了，特征提取网络不发生改变
    #       占用的显存较小，仅对网络进行微调
    #   Init_Epoch  模型当前开始的训练世代，开始大于Freeze_Epoch ，如设置：
    #               Init_Epoch = 60 Freeze_Epoch = 50  UnFreeze_Epoch = 100
    #               会跳过冻结阶段，直接从60开始，并调整对应的学习率
    #               （断点训练时使用）
    #   Freeze_Epoch 模型冻结训练的Freeze_Epoch
    #                （当Freeze_Train = False时失效）
    #   Freeze_batch_size 模型冻结训练的batch_size
    #                （当Freeze_Train = False时失效）
    # -------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 100
    Freeze_batch_size = 8
    # -------------------------------------------------------#
    #   解冻阶段训参数
    #       此时模型的主干不被冻结了，特征提取网络会发生改变
    #       占用的显存很大，网络的参数都会发生变化
    #   UnFreeze_Epoch 模型总训练的epoch
    #                  SGD需要更长的时间收敛，因此设置较大的UnFreeze_Epoch
    #                  Adam可以使用较小的UnFreeze_Epoch
    #   Unfreeze_batch_size 模型解冻的batch_size
    # -------------------------------------------------------#
    UnFreeze_Epoch = 150
    Unfreeze_batch_size = 12
    # -------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干后解冻训练
    # -------------------------------------------------------#
    Freeze_Train = False

    pretrained = True #  是否对主干Backbone进行训练，不训练则直接加载model_path

    # 设置用到的显卡
    distributed = False  # 指定是否单卡训练
    sync_bn = False # 是否DDP模式多卡可用
    fp16 = False # 是否使用很合精度验证，可减少一半的显存，需要pytorch1.7.1以上

    #TODO Mosaic 数据增强的效果更好

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda" ,local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank= {rank} ,local_rank={local_rank}) training...")
            print("Gpu Device Count:" ,ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    shuffle = True if distributed else False
    label_smoothing = 0 # 标签平滑 一般0.01以下，如0.01 0.005
    Init_lr = 1e-2 #模型学习率
    Min_lr = Init_lr * 0.01
    lr_decay_type = 'cos' # 学习率下降的方式，有cos step
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    save_period = 10 #多少次epoch保存一次权值
    input_shape = [416, 416]
    anchors_mask = [[6,7,8] ,[3,4,5] ,[0,1,2]] #用于帮助代码找到对应的先验框，一般不修改

    model_path = 'E:/Torch/yolov4-pytorch-master/model_data/yolo4_weights.pth' # 训练好的权值路径，SOTA数据结果
    save_dir = 'logs' # 保存权值和日志文件
    eval_flag = True # 是否训练时评估，评估对象为验证集
    eval_period = 2 #10 # 多少次epoch评估一次，不建议频繁。获得验证集的mAP和get_map.py稍有不同,参数更加保守.
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

    # *********************************************************#
    ## yolo-conv2d-1
    # *********************************************************#
    model = YoloBody(anchors_mask ,num_classes ,pretrained=pretrained)
    if not pretrained:
        print("wight_init()")
        weight_init(model)
    if model_path != '':
        # 根据预训练的全职key和weight进行加载
        if local_rank == 0:
            print("LOAD weigth file {}".format(model_path))
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
        # 显示没有匹配上的权值
        if local_rank == 0:
            print("\nSucessful Load Key:",str(load_key)[:500] ,"...\nSucessful Load Key Num:",len(load_key))
            print("\nFail To Load Key:",str(no_load_key)[:500] ,"...\nFail To Key Num:",len(no_load_key))
            print("\n\033[1;33;44m温馨还是不温馨都要提示：head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    # --------------------------------#
    #  损失函数
    # --------------------------------#
    yolo_loss = YoloLoss(anchors ,num_classes ,input_shape
        ,Cuda
        ,anchors_mask,label_smoothing
        ,0.005 #focal_loss
        ,0.25 #focal_alpha
        ,2 #focal_gamme
        ,'ciou' #iou_type
    )
    # --------------------------------#
    #  记录Loss数据
    # --------------------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now() ,'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir ,"loss_" + str(time_str))
        loss_history = LossHistory(log_dir ,model ,input_shape=input_shape)
    else:
        loss_history = None

    model_train = model.train()

    # --------------------------------#
    #  torch1.2不支持amp ,建议使用torch1.7.1及以上正确使用fp16
    #  因此torch1.2这里显示"could not be resolve"
    # --------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    # --------------------------------#
    #  多卡同步bn
    # --------------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # --------------------------------#
            #  多卡平行运行
            # --------------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train ,device_ids=[local_rank] ,find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # *********************************************************#
    #  读取数据集对应的txt
    # *********************************************************#
    with open(train_annotation_path ,encoding = 'utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path ,encoding ='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path ,anchors_path=anchors_path ,anchors_mask = anchors_mask ,model_path = model_path ,input_shape = input_shape,\
            Init_Epoch = Init_Epoch ,UnFreeze_Epoch = UnFreeze_Epoch ,Freeze_batch_size = Freeze_batch_size ,Unfreeze_batch_size = Unfreeze_batch_size ,Freeze_Train = Freeze_Train,\
            Init_lr = Init_lr ,Min_lr = Min_lr ,optimizer_type = optimizer_type ,momentum = momentum ,lr_decay_type = lr_decay_type ,\
            save_period = save_period ,save_dir = save_dir ,num_workers = num_workers ,num_train = num_train , num_val = num_val ,lr_scheduler_fuc = lr_decay_type
        )

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
        lr_scheduler_func = get_lr_scheduler(lr_decay_type ,Init_lr_fit ,Min_lr_fit ,UnFreeze_Epoch)

        # 判断每一个世代的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集")

        # 构建数据集加载器
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback = EvalCallback(model ,input_shape ,anchors ,anchors_mask ,class_names ,num_classes ,val_lines ,log_dir ,Cuda ,\
                  eval_flag = eval_flag ,period = eval_period)
        else:
            eval_callback = None

        # --------------------------------#
        #  开始训练模型
        # --------------------------------#
        for epoch in range(Init_Epoch ,UnFreeze_Epoch):

            # --------------------------------#
            #  如果模型有冻结学习部分
            #  则解冻，并设置参数
            # --------------------------------#
            if epoch >= UnFreeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # 判断当前batch_size，自适应调整学习率
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type in ['adam' ,'adamw'] else 5e-2
                lr_limit_min = 3e-4 if optimizer_type in ['adam' ,'adamw'] else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr , lr_limit_min) ,lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr ,lr_limit_min * 1e-2) ,lr_limit_max *1e-2)
                # 获得学习率下降的公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type ,Init_lr_fit ,Min_lr_fit ,UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train//batch_size
                epoch_step_val = num_val//batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("训练中数据集小，无法进行训练，请扩充数据集")

                gen = DataLoader(train_dataset ,shuffle=shuffle ,batch_size=batch_size,num_workers=num_workers ,pin_memory=True,
                                 drop_last=True ,collate_fn=yolo_dataset_collate ,sampler=train_sampler)
                gen_val = DataLoader(val_dataset ,shuffle=shuffle ,batch_size=batch_size ,num_workers=num_workers ,pin_memory=True,
                                drop_last=True ,collate_fn=yolo_dataset_collate ,sampler=val_sampler)

                UnFreeze_flag = True

            # --------------------------------#
            #  当前epoch训练模型
            # --------------------------------#
            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch

            set_optimizer_lr(optimizer ,lr_scheduler_func ,epoch)

            fit_one_epoch(model_train ,model ,yolo_loss ,loss_history ,eval_callback ,optimizer ,epoch ,epoch_step ,epoch_step_val ,gen ,gen_val ,UnFreeze_Epoch ,False #Cuda
                          ,fp16
                          ,scaler ,save_period ,save_dir
                          ,0 #local_rank
            )

        # done for

        if local_rank == 0:
            loss_history.writer.close()

