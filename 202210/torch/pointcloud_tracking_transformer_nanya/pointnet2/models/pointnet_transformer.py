""""
    #TODO
    1. pointnet++ backbone
    2. Cuda code
    3. cross complier
    4. KITTI data format

"""
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModule
from pointnet2.utils import pointnet2_utils
from pointnet2.models.multihead_attention import MultiheadAttention
from pointnet2.models.trainsformer import TransformerEncoder ,TransformerDecoder

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

class PositionEmbeddingLearned(nn.Module):
    """
    Absokute pos embedding ,learned.
    """
    def __init__(self ,input_chennel=3 ,num_pos_feats = 256):
        super(PositionEmbeddingLearned, self).__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_chennel ,num_pos_feats ,kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats ,num_pos_feats ,kernel_size=1)
        )

    def forward(self ,xyz):
        xyz = xyz.transpose(1,2).contiguous()
        # BX3XN
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
class PointnetTransformerSiamese(nn.Module):
    def __init__(self ,input_channels=3 ,
                 use_xyz=True,
                 objective=False,
                 input_size=1024,
                 sample_method=None):
        super(PointnetTransformerSiamese, self).__init__()
        self.input_size = input_size

        sample_method = None #'ffps'
        vote_sample_method = None # 'ffps'
        first_sample_method = None
        self.sample_method = sample_method
        d_model = 256
        num_layers = 1
        self.with_pos_embed = True

        self.backbone_net = Pointnet_Backbone(
            input_channels,
            use_xyz,
            sample_method=sample_method,
            first_sample_method=first_sample_method
        )
        self.cosine = nn.CosineSimilarity(dim=1)
        self.mlp = pt_utils.SharedMLP([4+256 ,256 ,256 ,256] ,bn=True)
        self.FC_layer_cla = (
            pt_utils.Seq(256)
            .conv1d(256 ,bn=True)
            .conv1d(256 ,bn=True)
            .conv1d(1 ,activation=None)
        )
        self.fea_layer = (
            pt_utils.Seq(256)
            .conv1d(256, bn=True)
            .conv1d(256 ,activation=None)
        )
        self.vote_layer = (
            pt_utils.Seq(3+256)
            .conv1d(256 ,bn=True)
            .conv1d(256,bn=True)
            .conv1d(3+256 ,activation=None)
        )
        self.group5 = pointnet2_utils.QueryAndGroup(1.0 ,8 ,use_xyz=use_xyz)
        self.group3 = pointnet2_utils.QueryAndGroup(0.3 ,8 ,use_xyz=use_xyz)
        self.group1 = pointnet2_utils.QueryAndGroup(0.1 ,16 ,use_xyz=use_xyz)

        self.vote_aggreagetion = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp = [1+256 ,256 ,256 ,256],
            use_xyz=use_xyz,
            sample_method=vote_sample_method
        )
        self.num_proposal = input_channels //16 #64
        self.FC_proposal = (
            pt_utils.Seq(256+1+256 + 3 + 256 + 3) # + 128+3
            .conv1d(256 ,bn=True)
            .conv1d(256 ,bn=True)
            .conv1d(256 ,bn=True)
            .conv1d(256 ,bn=True)
            .conv1d(3+1+1 ,activation=None)
        )
        multiheaad_attn = MultiheadAttention(
            feature_dim = d_model,
            n_head = 1,
            key_feature_dim = 128
        )
        if self.with_pos_embed:
            encoder_pos_embed = PositionEmbeddingLearned(3 ,d_model)
            decoder_pos_embed = PositionEmbeddingLearned(3 ,d_model)
        else:
            encoder_pos_embed = None
            decoder_pos_embed = None

        self.encoder = TransformerEncoder(
            multihead_attn=multiheaad_attn,
            FFN = None,
            d_model=d_model,
            num_encoder_layer= num_layers,
            self_posembed= encoder_pos_embed
        )
        self.decoder = TransformerDecoder(
            multihead_attn=multiheaad_attn,
            FFN=None,
            d_model=d_model,
            num_decoder_layers=num_layers,
            self_posembed=decoder_pos_embed
        )
