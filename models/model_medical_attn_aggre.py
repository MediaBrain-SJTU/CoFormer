import torch
import torch.nn as nn
import easydict
import copy
import math
from torch.autograd import Variable
import torch.nn.functional as F
import random
from dgl.nn.pytorch.factory import KNNGraph
import dgl
import numpy as np
import os

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.embedding = nn.Sequential(nn.Linear(vocab,d_model),nn.ReLU(),nn.Linear(d_model,d_model))

    def forward(self, x):
        return self.embedding(x)


class VariateEncoding(nn.Module):
    def __init__(self,v_num,h_dim=32):
        super(VariateEncoding,self).__init__()
        self.embedding = nn.Embedding(v_num,h_dim)
        
    def forward(self,x):
        return self.embedding(x)

# Positional Encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1) # 增加维度
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model)) #相对位置公式
         
        pe[:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:, 1::2] = torch.cos(position * div_term)   #取偶数列
        pe = pe.unsqueeze(0)           # 增加维度
        self.register_buffer('pe', pe)
         
    def forward(self, x):
        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1])
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加
        x = x.reshape(last_shape)
        return self.dropout(x)

class PositionalEncoding_irregular_plus(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding_irregular, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
         
    def forward(self, x, time):
        # print(time.shape,x.shape)

        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1])
            time = time.reshape(-1,time.shape[-1])

        pe = torch.zeros(x.shape[0],time.shape[1], self.d_model).cuda()
        position = time.unsqueeze(2) # 增加维度
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).cuda()#相对位置公式
         
        pe[:,:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:,:, 1::2] = torch.cos(position * div_term)   #取偶数列

        x = x + Variable(pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加
        x = x.reshape(last_shape)
        return self.dropout(x)


class PositionalEncoding_irregular(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000 ):
        super(PositionalEncoding_irregular, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model,d_model)
         
    def forward(self, x, time):

        if len(x.shape) > 3:
            last_shape = x.shape
            x = x.reshape(-1,x.shape[-2],x.shape[-1])
            time = time.reshape(-1,time.shape[-1])

        pe = torch.zeros(x.shape[0],time.shape[1], self.d_model).to(x.device)
        position = time.unsqueeze(2) # 增加维度
        div_term = torch.exp(torch.arange(0., self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model)).to(x.device)#相对位置公式
         
        pe[:,:, 0::2] = torch.sin(position * div_term)   #取奇数列
        pe[:,:, 1::2] = torch.cos(position * div_term)   #取偶数列

        # x = x + Variable(pe[:, :x.size(1)], requires_grad=False) # embedding 与positional相加

        x = self.linear(torch.cat((x,Variable(pe[:, :x.size(1)], requires_grad=False)),dim=-1))
        x = x.reshape(last_shape)
        return self.dropout(x)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer1,layer2, N):
        super(Encoder, self).__init__()
        self.layer1s = clones(layer1, N//2)
        self.layer2s = clones(layer2, N//2)
        self.N = N

        self.norm = LayerNorm(layer1.size) #归一化

    def forward(self, x, mask, time):
        "Pass the input (and mask) through each layer in turn."
        for i in range(self.N):
            # print('x',x.shape)
            if i%2 == 0:
                x = self.layer1s[i//2](x, mask) #把序列跟mask输入到每层中
            else:
                x = self.layer2s[i//2](x, mask, time)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #sublayer 1 
        self.feed_forward = feed_forward #sublayer 2 
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #拷贝两个SublayerConnection，一个为了attention，一个是独自作为简单的神经网络
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # print('x',x.shape,'attn',self.self_attn(x,x,x,mask).shape)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #attention层 对应层1
        return self.sublayer[1](x, self.feed_forward) # 对应 层2


class EncoderLayer_GAT(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout,num_neighbors):
        super(EncoderLayer_GAT, self).__init__()
        self.self_attn = self_attn #sublayer 1 
        self.feed_forward = feed_forward #sublayer 2 
        self.sublayer = clones(SublayerConnection(size, dropout), 2) #拷贝两个SublayerConnection，一个为了attention，一个是独自作为简单的神经网络
        self.size = size
        self.num_neighbors = num_neighbors

    def forward(self, x, mask, time):
        "Follow Figure 1 (left) for connections."
        # print('x',x.shape,'attn',self.self_attn(x,x,x,mask).shape)
        graphs = build_graph(time,x.device,self.num_neighbors)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, graphs)) #attention层 对应层1
        return self.sublayer[1](x, self.feed_forward) # 对应 层2

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    
    if len(mask.shape)<4:
        mask_matrix = torch.zeros((query.shape[0]*query.shape[1],query.shape[3],query.shape[3]))
        mask = mask.reshape(-1)
        for i in range(mask.shape[0]):
            mask_matrix[i,:mask[i],:mask[i]] = 1
        mask = mask_matrix.reshape(query.shape[0],1,query.shape[1],query.shape[3],query.shape[3])

    mask = mask.transpose(1,2).to(query.device)
    # print('score',scores.shape,'mask',mask.shape)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10) #mask必须是一个ByteTensor 而且shape必须和 a一样 并且元素只能是 0或者1 ，是将 mask中为1的 元素所在的索引，在a中相同的的索引处替换为 value  ,mask value必须同为tensor
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, q_dim, k_dim,v_dim, d_model, dropout=0.1,dim = 'Time'):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # d_v=d_k=d_model/h 
        self.h = h # heads 的数目文中为8
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.linear_q = nn.Linear(q_dim,d_model)
        self.linear_k = nn.Linear(k_dim,d_model)
        self.linear_v = nn.Linear(v_dim,d_model)
        self.linear_out = nn.Linear(d_model,q_dim)
        self.attn = None  
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if self.dim != 'Time':
            query,key,value = [x.transpose(1,2) for x in (query,key,value)]
        
        dim_to_keep = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k 


        query = self.linear_q(query).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        key = self.linear_k(key).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)
        value = self.linear_v(value).view(nbatches,dim_to_keep,-1,self.h,self.d_k).transpose(2,3)

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask,# 进行attention
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(2, 3).contiguous() \
            .view(nbatches, dim_to_keep ,-1, self.h * self.d_k) # 还原序列[batch_size,len,d_model]
        if self.dim != 'Time':
            x = x.transpose(1,2)

        return self.linear_out(x)

class GATlayer(nn.Module):
    def __init__(self,in_dim,out_dim,num_heads):
        super(GATlayer,self).__init__()
        each_out_dim = out_dim//num_heads
        # print('each',each_out_dim,out_dim,num_heads)
        self.gatconv = dgl.nn.pytorch.conv.GATv2Conv(in_dim, each_out_dim, num_heads=num_heads)

    def forward(self, x, graph):
        last_shape = x.shape
        x = x.reshape(-1,x.shape[3])
        # print('shape1',x.shape)
        x = self.gatconv(graph, x)
        # print('shape2',x.shape)
        x = x.reshape(last_shape)
        return x

def build_graph(time,device,num_neighbors=11): #(batch,agent,time)

    vertex = time.reshape(time.shape[0],time.shape[1]*time.shape[2]).unsqueeze(2)

    KNN = KNNGraph(num_neighbors)
    graphs = KNN(vertex)
    # print(graphs.num_nodes())

    return graphs.to(device)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
 
    def forward(self, x):
        "Norm"
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
 
 
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        # print('sub',x.shape,self.dropout(sublayer(self.norm(x))).shape)
        return x + self.dropout(sublayer(self.norm(x))) #残差连接


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model,d_out, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, input_dim,d_model, class_num):
        #初始化函数的输入参数有两个，d_model代表词嵌入维度，vocab.size代表词表大小
        super(Generator, self).__init__()
        #首先就是使用nn中的预定义线性层进行实例化，得到一个对象self.proj等待使用
        #这个线性层的参数有两个，就是初始化函数传进来的两个参数：d_model，vocab_size
        self.linear1 = nn.Linear(input_dim,d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model,d_model)
        self.linear3 = nn.Linear(d_model,class_num)

    def forward(self, x):
        #前向逻辑函数中输入是上一层的输出张量x,在函数中，首先使用上一步得到的self.proj对x进行线性变化,然后使用F中已经实现的log_softmax进行softmax处理。
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    Base for this and many other models.
    """
    def __init__(self, encoder,src_pe
                ,generator,num_agents=23,num_neighbors=23,agent_encoding_dim=32,d_model=256,static_dim=6):
        #初始化函数中有5个参数，分别是编码器对象，解码器对象,源数据嵌入函数，目标数据嵌入函数，以及输出部分的类别生成器对象.
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder 
        self.src_pe = src_pe
        self.embeds = nn.ModuleList([nn.Linear(1,d_model) for _ in range(num_agents)])
        self.static_embed = nn.Linear(static_dim,d_model)
        self.relu = nn.ReLU()
        self.generator = generator    # output generation module
        self.variate_embedd = VariateEncoding(num_agents)

        self.agent_aggre = MultiHeadedAttention(8,d_model+agent_encoding_dim,d_model+agent_encoding_dim,d_model+agent_encoding_dim,d_model)

        
    def forward(self, data, time, mask, static):
        "Take in and process masked src and target sequences."
        #在forward函数中，有四个参数，source代表源数据，target代表目标数据,source_mask和target_mask代表对应的掩码张量,在函数中，将source source_mask传入编码函数，得到结果后与source_mask target 和target_mask一同传给解码函数
        # print('src',src.shape)
        #data (batch,agent,time,1)
        if len(mask.shape)<4:
            mask_matrix = torch.zeros((data.shape[0]*data.shape[1],data.shape[2],data.shape[2]))
            mask = mask.reshape(-1)
            for i in range(mask.shape[0]):
                mask_matrix[i,:mask[i],:mask[i]] = 1
            mask = mask_matrix.reshape(data.shape[0],data.shape[1],data.shape[2],data.shape[2])
        data = data.unsqueeze(3)
        data_embeds = []
        for sensor in range(data.shape[1]):
            data_embeds.append(self.embeds[sensor](data[:,sensor]))
        data_embeds = torch.stack(data_embeds,dim=1)
        data_embeds = self.relu(data_embeds)

        memory_ = self.encode(data_embeds, mask,time)
        #memory_(batch,agent,time,d_model)
        #mask(batch,agent,time,time)
        #time(batch,agent,time)

        time_mask = torch.where(time>=0,1,0)
        feature_mask = time_mask.unsqueeze(3).repeat(1,1,1,memory_.shape[-1])

        memory_ = memory_.masked_fill(feature_mask==0,0)
        # print('memory',torch.isnan(memory_).any())

        memory_sum = memory_.sum(2)
        feature_regular = feature_mask.sum(2)+1e-9
        agentwise_memory = memory_sum/feature_regular#(b,a,f)
        # print('agent',torch.isnan(agentwise_memory).any())
        agentwise_memory = agentwise_memory.unsqueeze(2)

        agentwise_memory = self.agent_aggre(agentwise_memory,memory_,memory_,mask[:,:,0:1,:])
        agentwise_memory = agentwise_memory.squeeze(2)
        
        agent_mask = time_mask.sum(2)
        agent_mask = agent_mask.masked_fill(agent_mask>0,1)

        agent_mask_fill = agent_mask.unsqueeze(2).repeat(1,1,agentwise_memory.shape[-1])
        agentwise_memory = agentwise_memory.masked_fill(agent_mask_fill==0,0)
        aggre_feature = agentwise_memory.sum(1)/(agent_mask.sum(1)+1e-9).unsqueeze(1)

        # print('aggre',torch.isnan(aggre_feature).any())
        static_feature = self.static_embed(static)
        # print('static',torch.isnan(static_feature).any())
        all_feature = torch.cat((aggre_feature,static_feature),dim=-1)

        result = self.generator(all_feature)

        return result
        

    def encode(self, src, src_mask,src_time):
        #编码函数，以source和source_mask为参数,使用src_embed对source做处理，然后和source_mask一起传给self.encoder
        src_embedds = self.src_pe(src,src_time)
        # src_embedds = src
        variate_num = src.shape[1]
        variate_input = torch.arange(variate_num).unsqueeze(0).to(src.device)
        variate_embedd = self.variate_embedd(variate_input).unsqueeze(2)

        variate_embedd = variate_embedd.repeat(src.shape[0],1,src.shape[2],1)
        src_embedds = torch.cat((src_embedds,variate_embedd),dim=-1)


        return self.encoder(src_embedds, src_mask,src_time)




def make_model(src_vocab, tgt_vocab, N=8, d_model=256, d_ff=1024, h=8, dropout=0.1,num_agents=23,num_neighbors=23,agent_encoding_dim=32,static_dim=6):
    """
    构建模型
    params:
        src_vocab:
        tgt_vocab:
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model+agent_encoding_dim,d_model+agent_encoding_dim,d_model+agent_encoding_dim,d_model)
    # attn_a = MultiHeadedAttention(h,d_model,dim='Agent')
    attn_a = GATlayer(d_model+agent_encoding_dim,d_model+agent_encoding_dim,h)
    # attn_a = MultiHeadedAttention(h,d_model)
    ff = PositionwiseFeedForward(d_model+agent_encoding_dim, d_model+agent_encoding_dim, d_ff,dropout)
    position = PositionalEncoding_irregular(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model+agent_encoding_dim, c(attn), c(ff), dropout),EncoderLayer_GAT(d_model+agent_encoding_dim, c(attn_a), c(ff), dropout,num_neighbors), N),
        c(position),
        Generator(d_model+agent_encoding_dim+d_model,d_model, tgt_vocab),
        num_agents=num_agents,
        num_neighbors=num_neighbors,
        agent_encoding_dim=agent_encoding_dim,
        d_model=d_model,
        static_dim=static_dim)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


