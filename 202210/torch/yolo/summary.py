# *********************************************************#
#   用来查看网络结构
# *********************************************************#
import torch
from thop import clever_format ,profile
from torchsummary import summary

from net.yolo import YoloBody

if __name__ == "__main__":
    input_shape = [416 ,416]
    anchors_mask = [[6,7,8] ,[3,4,5] ,[0,1,2]]
    num_classes = 80

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = YoloBody(anchors_mask ,num_classes).to(device)
    summary(m ,(3 ,input_shape[0] ,input_shape[1]))

    dummy_input = torch.randn(1,3 ,input_shape[0] ,input_shape[1]).to(device)
    flops ,params = profile(m.to(device) ,(dummy_input ,) ,verbose=False)

    #---------------------------------------------------------#
    #  flops * 2 因为profile未将卷积作为两个operations
    #  有些论文 * 2 .有些论文未 * 2
    #  本代码选择乘2 ，参考YOLOX
    #---------------------------------------------------------#
    flops = flops * 2
    flops , params = clever_format([flops ,params] ,"%.3f")
    print("Total GFLOPS: %s" % (flops))
    print("Total params: %s" % (params))
