import torch.nn as nn
from pointnet2.utils import pointnet2_utils

class _PointnetSAModuleBase(nn.Module):
    def __init__(self ,sample_method=None):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.sample_method = sample_method


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    """
    Pointnet set abstraction layer with multiscale grouping

    :parameters
    npoint: int
            Number of features
    radii: list of float32
            list of radii to group with
    nsamples : list of int32
            Number of samples in each ball query
    mlps:
            Spec fo the pointnet before the global max_pool for each scale
    bn: bool
            Use batchnorm
    """
    def __init__(self ,radii ,nsamples ,mlps ,bn=True,
                 use_xyz=True ,vote=False ,sample_method=None):
        # tyoe : (PointnetSAModuleMSG , int , list[float] ,L=list[int] ,list[list[int] ,bool ,bool]) -> None
        super(PointnetSAModuleMSG ,self).__init__(sample_method=sample_method)
        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsamples = nsamples[i]
            if vote is False:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup(radius ,nsamples ,use_xyz = use_xyz)
                )
            else:
                self.groupers.append(
                    pointnet2_utils.QueryAndGroup_score(radius ,nsamples ,use_xyz = use_xyz)
                )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            self.mlps.append(pt_utils.ShareMLP(mlp_spec ,bn=bn)) #TODO replace this


class PointnetSAModule(PointnetSAModuleMSG):
    r""""
    Pointnet set abstrction layer

    :parameter
    npoint : int
            Number of features
    radius : float
            Radius of ball
    nsample : int
            Number of samples in the ball query
    mlp : list
            Spec of the pointnet before the global max_pool
    bn : bool
            Use batchnorm
    """

    def __init__(self ,mlp ,radius=None ,nsample=None ,bn=True ,use_xyz=True ,sample_method=None):
        # type: (PointnetSAModule ,list[int] ,int ,float ,int ,bool ,bool) -> None
        super(PointnetSAModule ,self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            sample_method=sample_method
        )