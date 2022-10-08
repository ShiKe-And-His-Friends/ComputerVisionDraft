import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# *********************************************************#
# MISH function
#   Yolo在darknet53上修改loss
#       a = log(1+e^x)
#       mish(x) =  tanh(a) =sinh(a) / cosh(a) = (e^a - e^-a) / (e^a + e^-a)
#               = (e^log(1+e^x) - e^-log(1+e^x)) / (e^log(1+e^x) + e^-log(1+e^x))
# *********************************************************#
class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self ,x):
        return x * torch.tanh(F.softplus(x))

class BasicConv(nn.Module):
    def __init__(self ,in_channels ,out_channels ,kernel_size ,stride=1):
        super(BasicConv ,self).__init__()
        self.conv = nn.Conv2d(in_channels ,out_channels ,kernel_size ,stride ,kernel_size//2 ,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self ,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# *********************************************************#
# darknet53 的小残差边
# 用于内部叠放
# *********************************************************#
class Resblock(nn.Module):
    def __init__(self ,channels ,hidden_channels = None):
        super(Resblock ,self).__init__()
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels ,hidden_channels ,1),
            BasicConv(hidden_channels ,channels ,3)
        )
    def forward(self ,x):
        return  x + self.block(x)

# *********************************************************#
# darknet53的结构块res_block_moudle
# 先用ZeroPadding2D和一个步长2x2卷积进行宽高压缩
# 然后建立一个大的残差边shortconv，绕过很多其他的残差结构
# 主干部分对num_blocks进行循环，循环的就是残差结构
# 对整个CSPdarknet53的结构快，就是大残差边+内部许多小残差边
# *********************************************************#
class Resblock_body(nn.Module):
    def __init__(self ,in_channels ,out_channels,num_blocks ,first=False):
        super(Resblock_body ,self).__init__()
        # --------------------------------#
        #  步长为2x2的卷积块进行宽高压缩
        # --------------------------------#
        self.downsample_conv = BasicConv(in_channels ,out_channels ,3 ,stride=2)
        if first:
            # --------------------------------#
            #  建立大的残差边self.split_conv0，绕过很多残差结构
            # --------------------------------#
            self.split_conv0 = BasicConv(out_channels ,out_channels ,1)
            # --------------------------------#
            # 主干部分对num_block进行循环，循环内部是残差结构
            # --------------------------------#
            self.split_conv1 = BasicConv(out_channels ,out_channels ,1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels ,hidden_channels=out_channels//2),
                BasicConv(out_channels,out_channels,1)
            )
            self.concat_conv = BasicConv(out_channels*2 ,out_channels,1)
        else:
            # --------------------------------#
            # 建立的模式同上，注意残差块需要多处混合
            # --------------------------------#
            self.split_conv0 = BasicConv(out_channels,out_channels//2 ,1)

            self.split_conv1 = BasicConv(out_channels ,out_channels//2 ,1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2 ,out_channels//2 ,1)
            )
            self.concat_conv = BasicConv(out_channels ,out_channels ,1)

    def forward(self,x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        # --------------------------------#
        # 堆叠大的残差边
        # --------------------------------#
        x = torch.cat([x1 ,x0] ,dim = 1)
        # --------------------------------#
        # 最通道进行整合
        # --------------------------------#
        x = self.concat_conv(x)

# *********************************************************#
# CSPdarknet53 的主体部分
# 输入一张 [3X416X416] 的图片
# 输出三个feature层
# *********************************************************#
class CSPDarkNet(nn.Module):
    def __init__(self ,layer):
        super(CSPDarkNet ,self).__init__()
        self.inplanes = 32
        # [3 x 416 x 416] -> [32 x 416 x 416]
        self.conv1 = BasicConv(3 ,self.inplanes ,kernel_size = 3 ,stride = 1)
        self.feature_channels = [64 ,128 ,256 ,512 ,1024]
        self.stages = nn.ModuleList([
            # [32 x 416 x 416] -> [64 x 208 x 208]
            Resblock_body(self.inplanes ,self.feature_channels[0] ,layer[0] ,first=True),
            # [64 x 416 x 416] ->
            Resblock_body(self.feature_channels[0] ,self.feature_channels[1] ,layer[1] ,first=False),
            # [128 x 104 x 104] -> [256 x 52 x 52]
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layer[2], first=False),
            # [256 x 52 x 52] -> [512 x 26 x 26]
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layer[3], first=False),
            # [512 x 26 x 26] -> [1024 x 13 x 13]
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layer[4], first=False)
        ])
        self.num_features = 1
        for m in self.modules():
            # 权值初始化
            if isinstance(m ,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0 ,math.sqrt(2. / n))
            elif isinstance(m ,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self ,x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[0](out4)
        return out3 ,out4 ,out5

def darknet53(pretrained = False):
    model = CSPDarkNet([1 ,2 ,8 ,8 ,4])
    if pretrained:
        # windows harddisk
        model.load_state_dict(torch.load("E:/Torch/yolov4-pytorch-master/model_data/CSPdarknet53_backbone_weights.pth"))
    return model