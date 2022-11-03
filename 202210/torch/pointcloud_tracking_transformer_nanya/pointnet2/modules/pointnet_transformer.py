""""
    #TODO
    1. pointnet++ backbone
    2. Cuda code
    3. cross complier
    4. KITTI data format

"""
import torch
import torch.nn as nn
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

            )
        )
