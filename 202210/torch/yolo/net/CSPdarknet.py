import torch
import torch.nn as nn
import torch.nn.functional as F

# *********************************************************#
# Yolo在darknet53上修改loss function Mish
# a = log(1+e^x)
# mish(x) =  tanh(a) =sinh(a) / cosh(a) = (e^a - e^-a) / (e^a + e^-a)
#           = (e^log(1+e^x) - e^-log(1+e^x)) / (e^log(1+e^x) + e^-log(1+e^x))
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
# darknet53的结构块res_block_moudle
# 先用ZeroPadding2D和一个步长2x2卷积进行宽高压缩
# 然后建立一个大的残差边shortconv，绕过很多其他的残差结构
# 主干部分对num_blocks进行循环，循环的就是残差结构
# 对整个CSPdarknet53的结构快，就是大残差边+内部许多小残差边
# *********************************************************#
class Resblock_body(nn.Module):
    def __init__(self ,in_channels ,out_channels,num_blocks ,first):
        super(Resblock_body ,self).__init__()
        # --------------------------------#
        #  步长为2x2的卷积块进行宽高压缩
        # --------------------------------#

# *********************************************************#
# CSPdarknet53 的主体部分
# 输入一张 [3X416X416] 的图片
# 输出三个feature层
# *********************************************************#
class CSPDarkNet(nn.Module):
    def __init__(self ,layer):
        super(CSPDarkNet ,self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3 ,self.inplanes ,kernel_size = 3 ,stride = 1)
        self.feature_channels = [64 ,128 ,256 ,512 ,1024]
        self.stages = nn.ModuleList([

        ])

def darknet53(pretrained = False):
    model = CSPDarkNet([1 ,2 ,8 ,8 ,4])
    if pretrained:
        # windows harddisk
        model.load_state_dict(torch.load("E:/Torch/yolov4-pytorch-master/model_data/CSPdarknet53_backbone_weights.pth"))
    return model