"""

    Transformer Moduel for PointCloud Tracking

    Thanks for project@PTTR that Nanyang Technology University shares codes

    2022-11-01

"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from torch import nn , Tensor
from .multihead_attention import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self ,d_model=512 ,nhead=1 ,num_layers=1 ,dim_feedforwards=2048,
                 activation="relu"):
        super(Transformer, self).__init__()
        multihead_attn = MulthiheadAttention(


        )
