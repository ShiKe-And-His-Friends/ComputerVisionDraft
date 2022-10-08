import torch
from torch import nn
from collections import OrderedDict
from net.CSPdarknet import darknet53

def conv2d(filter_in ,filter_out ,kernel_size ,stride =1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(filter_in ,filter_out ,kernel_size=kernel_size ,stride=stride ,padding=pad ,bias=False)),
        ("bn" ,nn.BatchNorm2d(filter_out)),
        ("relu" ,nn.LeakyReLU(0.1)),
    ]))

# *********************************************************#
# 三次卷积块
# *********************************************************#
def make_three_conv(filter_list ,in_filters):
    m = nn.Sequential(
        conv2d(in_filters ,filter_list[0] ,1),
        conv2d(filter_list[0] ,filter_list[1] ,3),
        conv2d(filter_list[1], filter_list[0], 1)
    )
    return m

def make_five_conv(filter_list ,in_filters):
    m = nn.Sequential(
        conv2d(in_filters ,filter_list[0] ,1),
        conv2d(filter_list[0] ,filter_list[1] ,3),
        conv2d(filter_list[1], filter_list[0], 1),
        conv2d(filter_list[0], filter_list[1], 3),
        conv2d(filter_list[1], filter_list[0], 1)
    )
    return m

# *********************************************************#
# SPP 结构，利用不同大小的池化核进行池化
# 池化后堆叠
# *********************************************************#
class SpatialPyramidPooling(nn.Module):
    def __init__(self ,pool_sizes=[5 , 9 ,13]):
        super(SpatialPyramidPooling ,self).__init__()
        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(pool_size ,1 , pool_size//2)
            for pool_size in pool_sizes
        ])
    def forward(self ,x):
        features = [maxpool(x) for maxpool in self.maxpools[:-1]]
        features = torch.cat(features + [x],dim =1)
        return features

# 卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self,in_channels ,out_channels):
        super(Upsample ,self).__init__()
        self.upsample = nn.Sequential(
            conv2d(in_channels ,out_channels ,1),
            nn.Upsample(scale_factor=2 ,mode='nearest')
        )

    def forward(self,x):
        x = self.upsample(x)
        return x

# 获得yolov4 的HEAD输出
def yolo_head(filters_list ,in_filters):
    m = nn.Sequential(
        conv2d(in_filters ,filters_list[0] ,3),
        nn.Conv2d(filters_list[0],filters_list[1] ,1)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self ,anchors_mask ,num_classes ,pretrained = False):
        super(YoloBody, self).__init__()
        # *********************************************************#
        #  backbone 使用 cs-darknet-53
        #  获得3个有效特征，分别是：
        #  [256 x 52 x 52]
        #  [512 x 26 x 26]
        #  [1024 x 13 x 13]
        # *********************************************************#
        self.backbone = darknet53(pretrained)
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes

        self.conv1 = make_three_conv([512,1024] ,1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512,1024] ,2048)

        self.upsample1 = Upsample(512 ,256)
        self.conv_for_P4 = conv2d(512 ,256 ,1)
        self.make_five_conv1 = make_five_conv([256 ,512] ,512)

        self.upsample2 = Upsample(256,128)
        self.conv_for_P3 = conv2d(256,128 ,1)
        self.make_five_conv2 = make_five_conv([128 ,256] ,256)

        # yolo HEAD3": 3*(5+num_classes) = 3*(5+20)= 3*(4+1+20) = 75
        self.yolo_head3 = yolo_head([256 ,len(self.anchors_mask[0]) * (5 + self.num_classes)] ,128)
        self.down_sample1 = conv2d(128 ,256 ,3 ,stride=2)
        self.make_five_conv3 = make_five_conv([256 ,512] ,512)

        # yolo HEAD2"
        self.yolo_head2 = yolo_head([512 ,len(self.anchors_mask[1]) * (5 + self.num_classes)] ,256)
        self.down_sample2 = conv2d(256 ,512 ,3 ,stride=2)
        self.make_five_conv4 = make_five_conv([512 ,1024] ,1024)

        # yolo HEAD1"
        self.yolo_head1 = yolo_head([1024 ,len(self.anchors_mask[2]) * (5 + self.num_classes)] ,512)

    def forward(self ,x):
        # backbone
        x2 ,x1 ,x0 = self.backbone(x)

        # [1024 x 13 x 13] -> [512 x 13 x 13] -> [1024 x 13 x 13] -> [512 x 13 x 13] -> [2048 x 13 x 13]
        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        # [2048 x 13 x 13] -> [512 x 13 x 13] -> [1024 x 13 x 13] -> [512 x 13 x 13]
        P5 = self.conv2(P5)

        # [512 x 13 x 13]->[256 x 13 x 13] -> [256 x 26 x 26]
        P5_upsample = self.upsample1(P5)
        # [512 x 26 x 26] -> [256 x 26 x 26]
        P4 = self.conv_for_P4(x1)
        # [256 x 26 x 26] + [256 x 26 x 26] -> [512 x 26 x 26]
        P4 = torch.cat([P4 ,P5_upsample] ,axis = 1)
        # [512 x 26 x 26] -> [256 x 26 x 26] -> [512 x 26 x 26] -> [256 x 26 x 26] -> [512 x 26 x 26] -> [256 x 26 x 26]

        P4 = self.make_five_conv1(P4)
        # [256 x 26 x 26] -> [128 x 26 x 26] -> [256 x 52 x 52]
        P4_upsample = self.upsample2(P4)
        # [256 x 52 x 52] -> [128 x 52 x 52]
        P3 = self.conv_for_P3(x2)
        # [128 x 52 x 52] + [128 x 52 x 52] -> [256 x 52 x 52]
        P3 = torch.cat([P3 ,P4_upsample] ,axis = 1)
        # [256 x 52 x 52] -> [128 x 52 x 52] -> [256 x 52 x 52] -> [128 x 52 x 52] -> [256 x 52 x 52] -> [128 x 52 x 52]
        P3 = self.make_five_conv2(P3)

        # [128 x 52 x 52]-> [256 x 26 x 26]
        P3_downsample = self.down_sample1(P3)
        # [256 x 26 x 26] + [256 x 26 x 26] -> [512 x 26 x 26]
        P4 = torch.cat([P3_downsample ,P4] ,axis =1)
        # [512 x 26 x 26] -> [256 x 26 x 26] -> [512 x 26 x 26] -> [256 x 26 x 26] -> [512 x 26 x 26] -> [256 x 26 x 26]
        P4 = self.make_five_conv3(P4)

        # [256 x 26 x 26] -> [512 x 13 x 13]
        P4_downsample = self.down_sample2(P4)
        # [512 x 13 x 13] + [512 x 13 x 13] -> [1024 x 13 x 13]
        P5 = torch.cat([P4_downsample ,P5] ,axis =1 )
        # [1024 x 13 x 13] -> [512 x 13 x 13] -> [1024 x 13 x 13] -> [512 x 13 x 13] -> [1024 x 13 x 13] -> [512 x 13 x 13]
        P5 = self.make_five_conv4(P5)

        # 第三个特征层
        out2 = self.yolo_head3(P3)

        # 第二个特征层
        out1 = self.yolo_head2(P4)

        # 第一个特征层
        out0 = self.yolo_head1(P5)

        # (batch_size ,75 ,13 ,13) (batch_size ,75 ,26 ,26) (batch_size ,75 ,52 ,52)
        return out0 ,out1 ,out2