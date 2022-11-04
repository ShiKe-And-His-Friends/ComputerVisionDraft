""""
    #TODO
    1. pointnet++ backbone
    2. Cuda code
    3. cross complier
    4. KITTI data format

"""
import torch
import torch.nn as nn
from pointnet2.utils.pointnet2_modules import PointnetSAModule

class Pointnet_Backbone(nn.Module):
    def __init__(self
        ,input_channels = 3,
        use_xyz = True,
        sample_method = None,
        first_sample_method = None
        ):
        super(Pointnet_Backbone ,self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius= 0.3,
                nsample=32,
                mlp=[input_channels ,64,64 ,128],
                use_xyz=use_xyz,
                sample_method=first_sample_method
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius= 0.5,
                nsample=32,
                mlp=[128 ,128, 128,256],
                use_xyz= use_xyz ,#False
                sample_method = sample_method
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256 ,256 ,256 ,256],
                use_xyz=use_xyz, #False
                sample_method=sample_method,
            )
        )
        self.cov_final = nn.Conv2d(256 ,256, kernel_size=1)
        self.sample_method = sample_method
