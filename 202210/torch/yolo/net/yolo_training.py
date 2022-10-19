import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self,anchors ,num_classes ,input_shape ,cuda ,anchors_mask = [[6,7,8] ,[3,4,5] ,[0,1,2]] ,label_smoothing = 0 ,focal_loss = False ,alpha = 0.25 ,gamma = 2 ,iou_type = 'ciou'):
        super(YOLOLoss ,self).__init__()
        # *********************************************************#
        #  13x13的特征层对应的anchor是[142 ,110] ,[192,243] ,[459 ,401]
        #  26x26的特征层对应的anchor是[36 ,75] ,[76,55] ,[72 ,146]
        #  52x52的特征层对应的anchor是[12 ,16] ,[19,36] ,[40 ,28]
        # *********************************************************#
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchor_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.balance = [0.4 ,1.0 ,4]
        self.box_ratio = 0.05
        self.obj_ratio = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2)
        self.cls_ratio = 1 * (num_classes / 80)

        self.focal_loss = focal_loss
        self.focal_loss_ratio = 10
        self.alpha = alpha
        self.gamma = gamma

        self.iou_type = iou_type
        self.ignore_threshold = 0.5
        self.cuda = cuda

    def clip_by_tensor(self ,t ,t_min ,t_max):
        t = t.float()
        result = (t > t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MEMLoss(self ,pred ,target):
        return torch.pow(pred - target ,2)

    def BCELoss(self ,pred ,target):
        epsilon = 1e-7
        pred = self.clip_by_tensor((pred ,epsilon ,1.0-epsilon))
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    # 平滑标签
    def smooth_labels(self ,y_true ,label_smoothing ,num_classes):
        return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

    def forward(self):
        return None

    def calculate_iou(self ,_box_a ,_box_b):
        # 计算真实框的左上角和右下角
        b1_x1 ,b1_x2 = _box_a[:,0] - _box_a[:,2]/2 , _box_a[:,0] + _box_a[:,2] /2
        b1_y1 ,b1_y2 = _box_a[:,1] - _box_a[:,3]/2 ,_box_a[: ,1] + _box_a[:,3] /2

        # 计算先验框获得预测框的左上角和右下角
        b2_x1 ,b2_x2 = _box_b[: ,0] - _box_b[:,2]/2 ,_box_b[:,0] + _box_b[:,2]/2
        b2_y1 ,b2_y2 = _box_b[:, 1] - _box_b[:,3]/2 ,_box_b[:,1] + _box_b[:,3]/2

        # 将真实框和预测框都转换为左上角右下角形式
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[: ,0] ,box_a[:,1] ,box_a[:,2] ,box_a[:,3] = b1_x1 ,b1_y1 ,b1_x2 ,b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        # 真实框和先验框的数量
        A = box_a.size(0)
        B = box_b.size(0)

        # 计算交的面积
        max_xy = torch.min(box_a[: ,2:].unsqueeze(1).expand(A ,B ,2) ,box_b[:,2:].unsqueeze(0).expand(A ,B ,2))
        min_xy = torch.max(box_a[:,:2].unsquenze(1).expand(A,B,2) ,box_b[:,:2].unsquenze(0).expand(A ,B ,2) )
        inter = torch.clamp((max_xy - min_xy) ,min =0)
        inter = inter[: ,: ,0] * inter[: ,: ,1]

        # 计算预测框和真实框各自的面积
        area_a = ((box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1])).unsqueeze(1).expand_as(inter) #[A ,B]
        area_b = ((box_b[:,2] - box_b[:,0]) * (box_b[:,3] - box_b[:,1])).unsqueeze(0).expand_as(inter) #[A ,B]

        # 求IOU
        union = area_a + area_b - inter
        return inter / union

    def get_target(self,l ,targets ,anchors ,in_h ,in_w):
        # 计算一共多少张图片
        bs = len(targets)
        # 用于选取哪些先验框不含物体
        noobj_mask = torch.ones(bs ,len(self.anchor_mask[1]) ,in_h ,in_w ,requires_grad=False)
        # 网络关注更小目标
        box_loss_scale = torch.zeros(bs ,len(self.anchor_mask[1]) ,in_h,in_w ,requires_grad=False)

        # batch_size,3 ,13 ,13 ,5 + num_classes
        y_true = torch.zeros(bs ,len(self.anchor_mask[1]) ,in_h ,in_w ,self.bbox_attrs,requires_grad=False)
        for b in range(bs):
            if len(targets[b]) == 0:
                continue
            batch_target = torch.zeros_like(targets[b])
            # 计算正样本在特征层的中心点
            batch_target[: ,[0,2]] = targets[b][: ,[0 ,2]] * in_w
            batch_target[: ,[1,3]] = targets[b][:,[1,3]] * in_h
            batch_target[: ,4] = targets[b][: ,4]

            # 将真实框转换成一个形式 num_true_box ,4
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0),2)), batch_target[: ,2:4]) ,1))
            # 将先验框转化成一个形式
            anchors_shapes = torch.FloatTensor(torch.cat((torch.zeros(len(anchors) ,2), torch.FloatTensor(anchors)),1))
            # 计算交并比
            best_ns = torch.argmax(self.calculate_iou(gt_box ,anchors_shapes) ,dim= -1)

            for t , best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[1]:
                    continue
                # 判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[l].index(best_n)
                # 获取真实框属于哪个网格点
                i = torch.floor(batch_target[t ,0]).long()
                j = torch.floor(batch_target[t ,1]).long()
                # 获取真实框的分类
                c = batch_target[t ,4].long()

                # noobj_mask 代表无目标的特征点
                noobj_mask[b ,k ,j ,i] = 0
                # tx ty 代表中心调整参数的真实值
                y_true[b , k ,j ,i ,0] = batch_target[t ,0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 4] = 1

                # 用于获取xywh的比例，大目标loss权值小/小目标loss权值大
                box_loss_scale[b , k ,j ,i] = batch_target[t ,2] * batch_target[t ,3] / in_h / in_w
        return y_true ,noobj_mask ,box_loss_scale

    def get_ignore(self ,l ,x ,y ,h ,w ,targets ,scaled_anchors ,in_h ,in_w ,noobj_mask):
        # 计算一共多少张图片
        bs = len(targets)
        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0 ,in_w -1 ,in_w).repeat(in_h ,1).repeat(
            int(bs * len(self.anchor_mask[l])) ,1 ,1).view(x.shape).type_as(x)
        grid_y = torch.linspace(0 ,in_h -1 ,in_h).repeat(in_w ,1).t().repeat(
            int(bs*len(self.anchor_mask[l])),1,1).view(y.shape).type_as(x)

        # 先验框的宽高
        scaled_anchors_1 = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = torch.Tensor(scaled_anchors_1).index_select(1 ,torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_1).index_select(1 ,torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(bs ,1).repeat(1 ,1 ,in_h * in_w).view(w.shape)
        anchor_w = anchor_h.repeat(bs ,1).repeat(1 ,1 ,in_h * in_w).wiew(h.shape)

        # 计算先验框调整后的中心和宽高
        pred_boxes_x = torch.unsqueeze(x + grid_x ,-1)
        pred_boxes_y = torch.unsqueeze(y + grid_y ,-1)
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w ,-1)
        pred_boxes_h = torch.unsqueeze(torch.exp(h)* anchor_h ,-1)
        pred_boxes = torch.cat([pred_boxes_x ,pred_boxes_y ,pred_boxes_w ,pred_boxes_h])

        for b in range(bs):
            # 预测结果转换一个形状
            pred_boxes_for_ignore = pred_boxes[b].view(-1 ,4)
            # 计算真实框，把真实框转换成相对特征层的大小
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                # 计算正样本在特征层上的中心点
                batch_target[: ,[0,2]] = targets[b][:,[0,2]] * in_w
                batch_target[: ,[1,3]] = targets[b][:,[1,3]] * in_h
                batch_target = batch_target[: , :4].type_as(x)

                # 计算交并比
                anch_ious =self.calculate_iou(batch_target ,pred_boxes_for_ignore)
                # 每个先验证框对应真实框的最大重合度 anch_ious_max  num_anchors
                anch_ious_max,_ = torch.max(anch_ious ,dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask ,pred_boxes

def weight_init(net ,init_type = 'normal' ,init_gain = 0.02):
    def init_func(m) :
        classname = m.__class__.__name__
        if hasattr(m ,'weight') and classname.find('Conv') != -1 :
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data ,0.0 ,init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data ,gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data ,a=0 ,mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data ,gain=init_gain)
            else:
                raise NotImplementedError('initialization method[%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data ,1.0 ,0.02)
            torch.nn.init.constant(m.bias.data ,0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func())
def get_lr_scheduler(lr_decay_type ,lr ,min_lr ,total_iters ,warmup_iters_ratio=0.5 ,warmup_lr_ratio=0.1 ,no_aug_iter_ratio = 0.05 ,step_num = 10):
    def yolox_warm_cos_lr(lr ,min_lr ,total_iters ,warmup_total_iters ,warmup_lr_start ,no_aug_iter,iters):
        if (iters <= warmup_total_iters):
            lr = (lr - warmup_lr_start) * pow(iter/float(warmup_total_iters) ,2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else :
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr ,decay_rate ,step_size ,iters):
        if (step_size < 1):
            raise ValueError("step_size must above 1")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if (lr_decay_type == 'cos'):
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters ,1) ,3)
        warmup_lr_start = max(warmup_lr_ratio * lr ,1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters ,1) ,15)
        func = partial(yolox_warm_cos_lr,lr ,min_lr ,total_iters ,warmup_total_iters ,warmup_lr_start ,no_aug_iter)
    else:
        decay_rate = (min_lr /lr) ** (1/(step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr ,decay_rate ,step_size)
    return func
def set_optimizer_lr(optimizer ,lr_scheduler_func ,epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
