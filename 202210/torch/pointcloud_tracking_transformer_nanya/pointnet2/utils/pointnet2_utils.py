"""
    南洋理工大学的实验室成员，一个私有库  git+https://github.com/v-wewei/etw_pytorch_utils.git@v1.1.1#egg=etw_pytorch_utils
    看起来是修改的程序配置的单例，不怎么必须
"""
from __future__ import(
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals
)
import torch
from torch.autograd import Function
import torch.nn as nn
import etw_pytorch_utils as pt_utils
import sys

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins ,"__POINTNET2_SETUP__" ,False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README:"
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst" # Thanks for your donate codes again
        )
if False:
    # Workaround for type hints without depending on the 'typing' module
    from typing import *
