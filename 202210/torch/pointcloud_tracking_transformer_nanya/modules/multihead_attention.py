import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self ,feature_dim=512 ,n_head=8 ,key_feature_dim=64,
                 extra_nonlinear=True):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        self.extra_nonlinear = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(Relati)

class RelationUnit(nn.Modle):
    def __init__(self ,feature_dim=512 ,key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 1
        self.WK = nn.Linear(feature_dim ,key_feature_dim ,bias=False)
        self.WQ = nn.Linear(feature_dim ,key_feature_dim ,bias=False)
        self.WV = nn.Linear(feature_dim ,key_feature_dim ,bias=False)
        self.after_norm = nn.BatchNorm1d(feature_dim)
        self.trans_conv = nn.Linear(feature_dim ,feature_dim ,bias=False)

        # init weights
        for m in 