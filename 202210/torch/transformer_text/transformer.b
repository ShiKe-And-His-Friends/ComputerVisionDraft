#
# https://wmathor.com/index.php/archives/1455/

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data 

# S: symbol that show starting of decoding input
# E: symbol that show starting of decoding output
# P: symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input     dec_input       dec_output
    ['inc mochte ein bier P' , 'S i want a beer .' ,'i want a beer . E'],
    ['inc mochte ein cola P' , 'S i want a coke .' ,'i want a coke . E'],
]

#padding should be zero
src_vocab = {'P' : 0 ,'ich':1 ,'mochte':2 ,'ein':3 ,'bier':4 ,'cola':5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0 ,'ich':1 ,'mochte':2 ,'ein':3 ,'bier':4 ,'cola':5 ,'S':6,'E':7 ,'.':8}
idx2word = {i: w for i ,w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length

def make_data(sentences):
    enc_inputs ,dec_inputs ,dec_outputs = [] ,[] ,[]
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] #[1,2,3,4,0] ,[1,2,3,5,0]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] #[6,1,2,3,4,8] ,[6,1,2,3,5,8]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] #[1,2,3,4,8,7] ,[1,2,3,5,8,7]

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs) ,torch.LongTensor(dec_inputs) ,torch.LongTensor(dec_outputs)

enc_inputs , dec_inputs ,dec_outputs = make_data(sentences)

class MyDataSet(Data.Dataset):
    def __init__(self ,enc_inputs ,dec_inputs ,dec_outputs):
        super(MyDataSet ,self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx],self.dec_inputs[idx],self.dec_outputs[idx]
    
loader = Data.DataLoader(MyDataSet(enc_inputs ,dec_inputs ,dec_outputs) ,2 ,True)

# 字嵌入 & 位置嵌入的维度，这两值是相同的，因此用一个变量就可以
d_model = 512 # Embedding Size
# FeedForward层隐藏神经元个数
d_ff = 2048 # FeedForward dimension
# Q K V 向量的维度，其中Q和K的维度必须一致，V的维度没有限制，不过为了方便起见，设置64
d_k = d_v = 64 # dimension of k(=Q) , V
# Encoder和Decoder的个数
n_layer  = 6 # number of Encoder of Decoder layer
# 多头注意力的head的数量
n_head =
