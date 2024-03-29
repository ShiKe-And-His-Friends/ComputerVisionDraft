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

from typing import Tuple, Union, Any

import torch
from torch import Tensor
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

class RandomDropout(nn.Module):
    def __init__(self ,p=0.5 ,inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self ,X):
        theta = torch.Tensor(1).uniform_(0 ,self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X ,theta ,self.train ,self.inplace)

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx ,xyz ,npoint):
        # type: (Any , torch.Tensor ,int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ------------------
        xyz: torch.Tensor
                (B N 3) tensor where N > npoint
        npoint: int32
                number of features in the sampled set
        Return:
        -----------------
                torch.Tensor
                (B npoint) tensor containing the set
        """
        # return _ext.furthest_point_sampling(xyz,npoint)
        fps_inds = _ext.furthest_point_sampling(xyz ,npoint)
        ctx.mark_non_differentiable(fps_inds)
        return  fps_inds

    @staticmethod
    def backward(xyz ,a=None):
        return None ,None

furthest_point_sample = FurthestPointSampling.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx ,features ,idx):
        # type(Any ,torch.Tensor ,torch.Tensor) -> torch.Tensor
        r"""
        Parameter
        -------------------
        features: torch.Tensor
                (B C N) tensor
        idx: torch.Tensor
                (B npoint) tensor of the features to gather
        Return:
        ----------------------
                torch.Tensor
                (B C npoint) tensor
        """
        _ , C ,N = features.size()
        ctx.for_backwards = (idx ,C ,N)
		ctx.mark_non_differentiable(idx)
        return _ext.gather_points(features ,idx)

    @staticmethod
    def backward(ctx ,grad_out):
        idx ,C ,N = ctx.for_backwards
        grad_features = _ext.gather_points_grad(grad_out.contiguous() ,idx ,N)
        return grad_features ,None

gather_operation = GatherOperation.apply
class ThreeNN(Function):
    @staticmethod
    def forward(ctx ,unknown ,known):
        # type (Any ,torch.Tensor ,torch.Tensor) -> Tuple[torch.Tensor ,torch.Tensor ,]
        r"""
        Find the three nearest neighbors of unknow in know
        Parameter
        ------------------
        unknown: torch.Tensor
                (B n 3) tensor of known features
        known: torch.Tensor
                (B m 3) tensor of unknown features
        Return:
        ------------------
        dist: torch.Tensor
                (B n 3) l2 distance to the three nearest neighbors
        idx: torch.Tensor
                (B n 3) index of 3 nearest neighbors
        """
        dist2 ,idx = _ext.three_nn(unknown ,known)
        ctx.mark_non_differentiable(idx)

        return torch.sqrt(dist2 ),idx

    @staticmethod
    def backward(ctx ,a=None ,b=None):
        return None ,None

three_nn = ThreeNN.apply

class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx ,features ,idx ,weight):
        # type (Any ,torch.Tensor ,torch.Tensor ,torch.Tensor) -> torch.Tensor
        r"""
        Performs weight linear interpolation on 3 features

        Parameter
        -------------------
        features: torch.Tensor
                (B c m) Features descriptors to be interpolated from
        idx: torch.Tensor
                (B n 3) three nearest neighbors of the target features in features
        weight: torch.Tensor
                (B n 3) weights
        Return:
        -------------------
                (B c n) tensor of the interpolated features
        """
        B ,c ,m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx , weight ,m)
        ctx.mark_non_differentiable(idx)

        return _ext.three_interpolate(features, idx ,weight)
    @staticmethod
    def backward(ctx ,grad_out):
        # type: (Any ,torch.Tensor) -> Tuple[torch.Tensor ,torch.Tensor ,torch.Tensor]
        idx ,weight , m= ctx.three_interpolate_for_backward
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous() ,idx ,weight ,m
        )
        return grad_features ,None ,None

three_interpolate = ThreeInterpolate.apply
class GroupingOperation(Function):
    @staticmethod
    def forward(ctx ,features ,idx):
        # type : (Any ,torch.Tensor ,torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ---------------
        features: torch.Tensor
                (B C N) tensor of features to group
        idx: torch.Tensor
                (B npoint nsample) tensor containing the indicies  of features to grouo with
        Return:
        ---------------
                torch.Tensor
                (B C npoints nsample) tensor
        """
        B ,nfeatures ,nsample = idx.size()
        _ ,C ,N = features.size()

        ctx.for_backwards = (idx,N)
        ctx.mark_non_differentiable(idx)

        return _ext.group_points(features ,idx)

    @staticmethod
    def backward(ctx ,grad_out):
        # type : (Any , torch.Tensor) -> Tuple[torch.Tensor ,torch.Tensor]
        r"""
        Parameters:
        ----------------
        grad_out : torch.Tensor
                (B C npoint nsample) tensor of the gradients of the output from forawrd

        Returns
        ----------------
                torch.Tensor
                (B C N) gradient of the features
        None
        """
        idx , N  = ctx.for_backwards
        grad_features = _ext.group_points_grad(grad_out.contiguous() , idx ,N)
        return grad_features ,None

grouping_operation = GroupingOperation.apply
class BallQuery(Function):
    @staticmethod
    def forward(ctx , radius ,nsample , xyz ,new_xyz):
        # type: (Any , float ,int ,torch.Tensor ,torch.Tensor) -> torch.tensor
        r"""
        Parameters
        -----------------
        radius: float
                 radius of the balls
        nsample: int
                maximum number of features in the balls
        xyz: torch.Tensor
                (B N 3) xyz coordinates of the features
        new_xyz: torch.Tensor
                (B npoint 3) centers of the ball query

        Return
        -----------------
                torch.tensor
                (B npoint nsample) tensor with the indicies of the features that from the query balls
        """
        idxs = _ext.ball_query(new_xyz ,xyz ,radius ,nsample)
        ctx.mark_non_differentiable(idxs)
        return idxs

    @staticmethod
    def backward(ctx, a=None):
        return None ,None ,None ,None

ball_query = BallQuery.apply
class BallQuery_score(Function):
    @staticmethod
    def forward(ctx ,radius ,nsample ,xyz ,new_xyz ,score):
        # type: (Any ,float ,int ,torch.Tensor ,torch.Tensor) -> torch.Tensor
        r"""
        Parameter:
        ---------------
        radius: float
                radius of the balls
        nsample: int
                maximum number of features in the balls
        xyz: torch.Tensor
                (B N 3) xyz coordinates of the features
        new_xyz: torch.Tensor
                (B npoints 3)  centers of the ball query
        score: float
            score numbers

        Return:
        ---------------
                torch.Tensor
                (B npoint nsample) tensor with the indicies of the features that from the query balls
        """
        # return _ext.ball_query_score(new_xyz ,xyz ,score ,radius ,nsample)
        inds = _ext.ball_query_score(new_xyz ,xyz ,score ,radius ,nsample)
        ctx.mark_non_differentiable(inds)
        return inds

    @staticmethod
    def backward(ctx ,a=None):
        return None ,None ,None ,None ,None

ball_query_score = BallQuery_score.apply

class QueryAndGroup(nn.Module):
    r"""
    Group with a ball query of radius

    :parameter
    -------------
    radius : float 32
            Radius of ball
    nsamples : int32
            Maximum number of features to gather in the ball
    """
    def __init__(self ,radius ,nsample ,use_xyz=True):
        # type:(QueryAndGroup ,float ,int ,bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius ,self.nsample ,self.use_xyz = radius ,nsample ,use_xyz

    def forward(self ,xyz ,new_xyz ,features=None):
        # type:(QueryAndGroup ,torch.Tensor ,torch.Tensor ,torch.Tensor ) -> Tuple[torch.Tensor]
        r"""
        :parameters
        ---------------
        xyz: torch.Tensor
                xyz coordinates of the features (B N 3)
        new_xyz: torch.Tensor
                centriods (B npoint, 3)
        features: torch.Tensor
                Descriptors of the features (B C N)

        :return:
        ----------------
        new_features : torch.Tensor
                (B ,3+C ,npoints , nsample) tensor
        """
        idx = ball_query(self.radius ,self.nsample ,xyz ,new_xyz)
        xyz_trans = xyz.transpose(1 ,2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans,idx) # (B ,3 ,npoint ,nsample)
        grouped_xyz -= new_xyz.transpose(1 ,2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features ,idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz ,grouped_features] , dim=1
                ) # (B C+3 npoints nsample)
            else :
                new_features = grouped_features
        else :
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features

class QueryAndGroup_score(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameter:
    --------------
    radius : float32
            Radius of ball
    nsamples: int32
            Maximum number of features to gather in the ball
    """

    def __init__(self ,radius ,nsample ,use_xyz=True):
        # type: (QueryAndGroup_score ,float ,int ,bool) -> None
        super(QueryAndGroup_score ,self).__init__()
        self.radius ,self.nsample ,self.use_xyz = radius,nsample ,use_xyz

    def forward(self ,xyz ,new_xyz ,score ,features = None):
        # type:(QueryAndGroup_score ,torch.Tensor ,float ,torch.Tensor ,torch.Tensor) -> Tuple[Union[Tensor, Any], Any]
        r"""
        Parameters
        -----------------
        xyz: torch.Tensor
                xyz coordinates of the features (B N 3)
        new_xyz: torch.Tensor
                centriods (B npoint 3)
        features: torch.Tensor
                Descriptors of the features (B C N)
        Return:
        ----------------
        new_features : torch.Tensor
                (B 3+C npoint nsample) tensor
        """
        idx = ball_query(self.radius ,self.nsample ,xyz ,new_xyz)
        unique_score = ball_query_score(self.radius ,self.nsample ,xyz ,new_xyz ,score)
        score_id = unique_score.argmax(dim=1)

        xyz_trans = xyz.transpose(1 ,2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans ,idx) # (B 3 npoints nsample)
        grouped_xyz -= new_xyz.transpose(1,2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features ,idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz ,grouped_features] ,dim =1
                ) # (B C+3 npoint ,nsample)
            else:
                new_features = grouped_features
        else:
            assert(
                self.use_xyz
            ) ,"Cannot have not features and not use xyz as a features!"
            new_features = grouped_xyz
        return new_features,score_id


class GroupAll(nn.Module):
    r"""
    Groups all features
    """
    def __init__(self ,use_xyz = True):
        # type: (GroupAll ,bool) -> None
        super(GroupAll ,self).__init__()
        self.use_xyz = use_xyz

    def forward(self ,xyz ,new_xyz ,features=None):
        # type (GroupAll ,torch.Tensor ,torch.Tensor ,torch.Tensor) -> Tuple[torch.Tensor]
        r"""
        Parameter:
        ----------------------
        xyz: torch.Tensor
                xyz coordinates of the features (B N 3)
        new_xyz: torch.Tensor
                Ignored
        features: torch.Tensor
                Descriptors of the features (B C N)
        Return:
        ----------------------
                new_features : torch.Tensor
                (B C+3 1 N) tensor
        """
        grouped_xyz = xyz.transpose(1,2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz ,grouped_features] ,dim=1
                ) # (B 3+C 1 N)
            else :
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features




