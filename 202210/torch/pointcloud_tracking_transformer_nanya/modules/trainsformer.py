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
from typing import Optional ,List
from .multihead_attention import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self ,d_model=512 ,nhead=1 ,num_layers=1 ,dim_feedforwards=2048,
                 activation="relu"):
        super(Transformer, self).__init__()
        multihead_attn = MulthiheadAttention(
            feature_dim = d_model,
            n_head = 1,
            key_feature_dim = 128
        )
        #FFN_conv = nn.Conv2d() # do not use feed-forward network
        self.encoder = TransformerEncoder(
            multihead_attn = multihead_attn,
            FFN=None,
            d_model = d_model,
            num_encoder_layer= num_layers
        )
        self.decoder = TransformerDecoder(
            multihead_attn = multihead_attn,
            FFN = None,
            d_model = d_model,
            num_decoder_layers = num_layers
        )

    def forward(self ,feature):
        num_img_train = feature.shape[0]

        ## encoder
        encoder_memory,_ = self.encoder(feature)

        ## decoder
        for i in range(num_img_train):
            _ ,cur_encoded_feat = self.decoder(
                feature[i ,...].unsqueeze(0),
                memory= encoder_memory
            )
            if i == 0 :
                encoded_feat = cur_encoded_feat
            else:
                encoded_feat = torch.cat((encoded_feat ,cur_encoded_feat) ,0)
        return encoded_feat ,decoded_feat

class TransformerEncoderLayer(nn.Module):
    def __init__(self ,multihead_attn ,FFN ,d_model ,self_posembed=None):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = multihead_attn
        # Implement of Feadforward model
        self.FFN = FFN
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed
        self.dropout = nn.Dropout(0.1)

    def with_pos_embed(self ,tensor ,pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self ,src ,query_pos = None):
        # BxNxC -> BxCxN -> NxBxC
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2 ,0 ,1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src ,query_pos_embed)

        # self-attention
        # NxBxC
        src2 = self.self_attn(query=query ,key=key ,value=value)
        src = src + src2

        # NxBxC -> BxCxN -> NxBxC
        src = self.norm(src.permute(1,2,0)).permute(2,0,1)

        return F.relu(src)


class TransformerEncoder(nn.Module):
    def __init__(self ,multihead_attn ,FFN ,\
                 d_model = 512, num_encoder_layer=6 , activation="relu",\
                 self_posembed=None):
        super(TransformerEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn ,FFN ,d_model ,self_posembed=self_posembed
        )
        self.layers = _get_clones(encoder_layer ,num_encoder_layer)

    def forward(self ,src ,query_pos=None):
        num_imgs ,batch ,dim = src.shape
        output = src

        for layer in self.layers:
            output = layer(output ,query_pos=query_pos)

        # import pdb ; pdb.set_trace()
        # [L B D] -> [B D L]
        # output_feat = output.reshape(num_imgs , bathc ,dim)
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self ,multihead_attn ,FFN ,d_model ,self_posembed=None):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(
            feature_dim=d_model,
            n_head=1,
            key_feature_dim=128
        )
        self.FFN = FFN
        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def with_pos_embed(self ,tensor ,pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self ,tgt , memory ,query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2 ,0 ,1)
        else:
            query_pos_embed = None
        # NxBxC

        # self-attention
        query = key = value = self.with_pos_embed(tgt ,query_pos_embed)

        tgt2 = self.self_attn(query=query ,key= key ,value=value)
        # tgt2 = self.droupout1(tgt2)
        tgt = tgt + tgt2
        # tgt = F.relu(tgt)
        # tgt = self.instance_norm(tgt ,input_shape)
        # NxBxC
        # tgt = self.norm(tgt)
        tgt = self.norm1(tgt.permute(1 ,2 ,0)).permute(2 ,0 ,1)
        tgt = F.relu(tgt)

        mask = self.cross_attn(query=tgt ,key=memory ,value = memory)
        # mask = self.droupout2(mask)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1 ,2 ,0)).permute(2 ,0 ,1)

        tgt2 = F.relu(tgt2)
        return tgt2


class TransformerDecoder(nn.Module):
    def __init__(self ,multihead_attn ,FFN,
                 d_model = 512,
                 num_decoder_layers = 6,
                 activation="relu",
                 self_posembed = None):
        super(TransformerDecoder, self).__init__()
        decoder_layer = TransformerDecoderLayer(
            multihead_attn ,FFN,
            d_model,self_posembed=self_posembed)
        self.layers = _get_clone(decoder_layer ,num_decoder_layers)

    def forward(self ,tgt ,memory ,query_pos=None):
        assert tgt.dim() == 3 ,'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs , batch ,dim = tgt.shape

        output = tgt
        for layer in self.layers:
            output = layer(output ,memory ,query_pos = query_pos)
        return output

def _get_clone(module ,N):
    return nn.ModuleList([module for i in range(N)])
