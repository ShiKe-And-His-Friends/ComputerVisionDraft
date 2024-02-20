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
    ['ich mochte ein bier P' , 'S i want a beer .' ,'i want a beer . E'],
    ['ich mochte ein cola P' , 'S i want a coke .' ,'i want a coke . E'],
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
n_layers  = 6 # number of Encoder of Decoder layer
# 多头注意力的head的数量
n_heads = 8 # number of heads in Multi-Head Attention

# 有些函数类Encoder和Decoder都会调用，不确定是src_len还是tgt_len，对不确定的用seq_len

#
# Positional Encoding 位置编码
class PositionalEncoding(nn.Modul):
    def __init__(self ,d_model ,dropout=0.1 ,max_len=5000):
        super(PositionalEncoding ,self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len ,d_model)
        position = torch.arange(0,max_len ,dtype=torch.float).unsqueeze(1)
        div_item = torch.exp(torch.arange(0,d_model,2).float() * (-math.long(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_item)
        pe[:,1::2] = torch.cos(position * div_item)
        pe = pe.unsqueeze(0).transpose(0 ,1)
        self.register_buffer('pe',pe)

    def forward(self ,x):
        '''
        x : [seq_len ,batch_size ,d_model]
        '''
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)

#
# Pad Mask 针对句子不够长，增加pad，需要对pad进行mask
def get_attn_pad_mask(seq_q ,seq_k):
    '''
    seq_q : [batch_size ,seq_len]
    seq_k : [batch_size ,seq_len]
    seq_len could be src_len or it cloud be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size , len_q = seq_q.size()
    batch_size , len_k = seq_k.size()
    #eq(zero) is PAD token
    # 函数核心是seq_k.data.eq(0) 返回一个大小和seq_k一样的tensor,只不过里面只有True和False
    pad_attn_mask = seq_k.data.eq(0).unsquenze(1) #[batch_size ,1 ,len_k],True is masked
    return pad_attn_mask.expand(batch_size ,len_q ,len_k) # [batch_size ,len_q ,len_k]


#
# Subsequence Mask ,Decoder input 不希望看到未来时刻的单词信息，需要mask
def get_attn_subsequence_mask(seq):
    '''
    seq : [batch_size ,tgt_len]
    '''
    attn_shape = [seq.size(0) ,seq.size(1) ,seq.size(1)]
    # 只有Decoder会用到，numpy生成一个全1的方阵，通过triu()生成上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape),k =1) #Upper trigangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask #[batch_size ,tgt_len ,tgt_len]

#
# ScaledDotProductAttention ,计算context vector
class ScaledDotProductAttention(nn.Model):
    def __init__(self):
        super(ScaledDotProductAttention ,self).__init__()

    def forward(self ,Q ,K ,V ,attn_mask):
        '''
        Q : [batch_size ,n_heads ,len_q ,d_k]
        K : [batch_size ,n_heads ,len_k ,d_k]
        V : [batch_size ,n_heads ,len_v(=len_k) ,d_v]
        '''
        
        # 将需要屏蔽的信息屏蔽掉
        scores = torch.matmul(Q ,K.transpose(-1,-2)) / np.sqrt(d_k) # socres : [batch_size ,n_heads ,len_q ,len_k]
        scores.masked_fill(attn_mask ,-1e9) # Fill elements of self tensor with value where mask is True
        
        # 将 Q 和 K 相乘计算scores ，然后scores和K相乘，得到每个单词的context vector
        attn = nn.Softmax(dim = -1)(scores)
        context = torch.matmul(attn ,V) #[batch_size ,n_heads ,len_q ,d_v]
        return context ,attn

#
# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_model ,d_k * n_heads ,bias=False)
        self.W_K = nn.Linear(d_model ,d_k * n_heads ,bias=False)
        self.W_V = nn.Linear(d_model ,d_v * n_heads ,bias=False)
        self.fc = nn.Linear(n_heads * d_v ,d_model ,bias=False)

    '''
    完整代码三处调用MultiHeadAttention，
    EncoderLayer调用一次，传入input_Q ,input_K ,input_V全部是enc_inputs
    DecoderLayer调用两次，第一次传入dec_inputs,第二次分别传入dec_outputs ,enc_outputs ,enc_outputs
    '''
    def forward(self ,input_Q ,input_K ,input_V ,attn_mask):
        '''
        input_Q : [batch_size ,len_q ,d_model]
        input_K : [batch_size ,len_k ,d_model]
        input_V : [batch_size ,len_v(=len_k),d_model]
        attn_mask: [batch_size ,seq_len ,seq_len]
        '''
        residual ,batch_size = input_Q ,input_Q.size(0)
        # (B ,S ,D) -proj -> (B ,S ,D_new) -spilt->(B ,S ,H ,W) -trans->(B ,H ,S ,W)
        Q = self.W_Q(input_Q).view(batch_size ,-1 ,n_heads ,d_k).transpose(1 ,2) # Q : [batch_size ,n_heads ,;en_q ,d_k]
        K = self.W_K(input_K).view(batch_size ,-1 ,n_heads ,d_k).transpose(1 ,2) # K : [batch_size ,n_heads ,len_k ,d_k]
        V = self.W_V(input_V).view(batch_size ,-1 ,n_heads ,d_v).transpose(1 ,2) # V : [batch_size ,n_heads ,len_v(=len_k),d_k]

        attn_mask = attn_mask.unsqueeze(1).repeat(1 ,n_heads ,1 ,1) # attn_mask : [batch_size ,n_heads ,seq_len ,len_v(len_k) ,d_v]

        # context : [batch_size ,n_heads ,len_q ,d_v]
        # attn : [batch_size ,n_heads ,len_q ,len_k]
        context ,attn = ScaledDotProductAttention()(Q ,K ,V ,attn_mask)
        # context : [batch_size ,len_q ,n_heads * d_v]
        context = context.transpose(1 ,2).reshape(batch_size ,-1 ,n_heads* d_v)
        output = self.fc(context) # [batch_size ,len_q ,d_model]
       
        return nn.LayerNorm(d_model).cuda()(output+residual) , attn

#
# FeedForward Layer
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet ,self).__init__()
        # 两次线性变换，残差连接后再做一个LayerNorm
        self.fc = nn.Sequential(
            nn.Linear(d_model ,d_ff ,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff ,d_model ,bias=False)
        )
    
    def forward(self ,inputs):
        '''
        inputs : [batch_size ,seq_len ,d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size ,seq_len ,d_model]

#
# Encoder Layer
class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer ,self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self ,enc_inputs ,enc_self_attn_mask):
        '''
        enc_inputs : [batch_size ,src_len ,d_model]
        enc_self_attn_mask : [batch_size ,src_len ,src_len]
        '''
        # enc_outputs : [batch_size ,src_len ,d_model]
        # attn : [batch_size ,n_heads ,src_len ,src_len]
        enc_outputs ,attn = self.enc_self_attn(enc_inputs ,enc_inputs ,enc_inputs ,enc_self_attn_mask)
        
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # [batch_size ,src_len ,d_model]

        return enc_outputs ,attn

#
# Encoder ,拼接Encoder Layer
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder ,self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size ,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([Encoderlayer() for _ in range(n_layers)])

    def forward(self ,enc_inputs):
        '''
        enc_inputs : [batch_size ,src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs) #[batch_size ,src_len ,d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0,1)).transpose(0,1) #[batch_size ,src_len ,src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs ,enc_inputs) #[batch_size ,src_len ,src_len]
        
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs:[natch_size ,src_len ,d_model] 
            # enc_self_attn:[batch_size ,n_heads ,src_len ,src_len]
            enc_outputs ,enc_self_attn = layer(enc_outputs ,enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        
        return enc_outputs , enc_self_attns


#
# Decoder Layer
class DecoderLayer(nn.Module):
    '''
    DecoderLayer 调用两次MultiHeadAttention
    第一次计算input的self-attention得到输出dec_outputs。将dec_output做为生成Q的元素，enc_output做为 生成K和V的元素
    再调用一次，得到Encoder 和 DecoderLayer之间的context vector
    最后将dec_outputs做一次维度变换，然后返回
    '''
    def __init__(self):
        super(DecoderLayer ,self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self ,dec_inputs ,enc_outputs ,dec_self_attn_mask ,dec_enc_attn_mask):
        '''
        dec_inputs : [batch_size ,tgt_len ,d_model]
        enc_outputs : [batch_size ,src_len ,d_model]
        dec_self_attn_mask : [batch_size ,tgt_len ,tgt_len]
        dec_enc_attn_mask : [batch_size ,tgt_len ,src_len]
        '''
        # dec_ouputs : [batch_size ,tgt_len ,d_model]
        # dec_self_attn : [batch_size ,n_heads ,tgt_len ,tgt_len]
        dec_outputs ,dec_self_attn = self.dec_self_attn(dec_inputs ,dec_inputs ,dec_inputs ,dec_self_attn_mask)
        
        # dec_outputs : [batch_size ,tgt_len ,d_model]
        # dec_enc_attn : [batch_size ,h_heads ,tgt_len ,src_len]
        dec_outputs,dec_enc_attn = self.dec_enc_attn(dec_outputs ,enc_outputs ,enc_outputs ,dec_enc_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs) #[batch_size ,tgt_len ,d_model]

        return dec_outputs ,dec_self_attn ,dec_enc_attn
    
#
# Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder ,self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size ,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self ,dec_inputs ,enc_inputs ,enc_outputs):
        '''
        dec_inputs : [batch_size ,tgt_len]
        enc_inputs : [batch_size ,src_len]
        enc_outputs : [batch_size ,src_len ,d_model]
        '''
        dec_outpust = self.tgt_emb(dec_inputs) #[batch_size ,tgt_len ,d_model]
        # [batch_size ,tgt_len ,d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1).cuda() 
        
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs ,dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        
        # Decoder不仅要mask掉pad信息，还要mask掉未来时刻的信息
        # torch.gt()将各个位置上元素和value比较，大于value置1，否则置0
        # 有以下三行

        # [batch_size ,tgt_len ,tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask) ,0).cuda() 

        # [batch_size ,tgt_len ,src_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs ,enc_inputs) 

        dec_self_attns ,dec_enc_attns = [],[]
        for layer in self.layers:
            # dec_outputs : [batch_size ,tgt_len ,d_model]
            # dec_self_attn : [batch_size ,n_heads ,tgt_len ,tgt_len]
            # dec_enc_attn : [batch_size ,h_heads ,tgt_len ,src_len]
            dec_outputs ,dec_self_attn ,dec_enc_attn = layer(dec_outputs ,enc_outputs ,dec_self_attn_mask ,dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs ,dec_self_attns ,dec_enc_attns
    
#
# 
