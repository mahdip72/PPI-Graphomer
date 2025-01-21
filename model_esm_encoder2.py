import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import esm
# Set CUDA devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_detail_save_dir="/public/mxp/xiejun/py_project/PPI_affinity/runs/run_9_pretrainedEsm/"
# ff = open(model_detail_save_dir + 'model_detail.txt', 'w')



class Config:
    def __init__(self, pro_vocab_size, device, pro_len, d_embed, d_ff, d_k, d_v, n_layers_en, n_heads):
        self.device = device
        self.pro_vocab_size = pro_vocab_size
        self.pro_len = pro_len
        self.d_embed = d_embed
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_layers_en = n_layers_en
        self.n_heads = n_heads

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.2, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.config.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)  # attn = self.dropout(F.softmax(attn, dim=-1))
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class ScaledDotProductAttention_bias(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention_bias, self).__init__()
        self.config = config
        self.scale_factor = nn.Parameter(torch.tensor([1.0]*config.n_heads))
    def forward(self, Q, K, V, attn_mask,attn_mask_if,ia_type,ia_feat,distance_matrix):
    # def forward(self, Q, K, V, attn_mask,attn_mask_if):
        scores1 = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.config.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        
        
        # ia_type = ia_type / math.sqrt(self.config.d_k)
        # ia_feat = ia_feat / math.sqrt(self.config.d_k)

        # scale=torch.mean(scores1.reshape(scores1.shape[0],-1),dim=1)
        # scale=self.scale_factor.to(self.config.device).unsqueeze(0).repeat(Q.shape[0],1)

        # bias=(ia_type+ia_feat)*distance_matrix.unsqueeze(1).expand(ia_type.shape[0], ia_type.shape[1], 2000, 2000)*scale.view(scale.shape[0], self.config.n_heads, 1,1)
        # bias=(ia_type+ia_feat)*distance_matrix.unsqueeze(1).expand(ia_type.shape[0], ia_type.shape[1], 2000, 2000)
        bias=(ia_type+ia_feat)


        bias.masked_fill_(attn_mask_if, -1e9)  # Fills elements of self tensor with value where mask is True.
        # ia_feat.masked_fill_(attn_mask_if, -1e9)  # Fills elements of self tensor with value where mask is True.
       
        scores=scores1+bias
        # scores=scores1

        scores.masked_fill_(attn_mask_if, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)  # attn = self.dropout(F.softmax(attn, dim=-1))
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.W_Q = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_embed, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.n_heads * config.d_v, config.d_embed, bias=False)
        self.ln = nn.LayerNorm(config.d_embed)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention(self.config)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.config.n_heads * self.config.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_embed]
        return self.ln(output + residual), attn

class MultiHeadAttention_bias(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention_bias, self).__init__()
        self.config = config
        self.W_Q = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_Q2 = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_K2 = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_embed, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.n_heads * config.d_v, config.d_embed, bias=False)
        self.ln = nn.LayerNorm(config.d_embed)

    def forward(self, input_Q, input_K, input_V, attn_mask,bias_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]


        Q2 = self.W_Q2(input_Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K2 = self.W_K2(input_K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        bias_mask = bias_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        context, attn_bias = ScaledDotProductAttention_bias(self.config)(Q, K, V,Q2,K2 ,attn_mask,bias_mask)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.config.n_heads * self.config.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_embed]
        return self.ln(output + residual), attn_bias

class MultiHeadAttention_Rope3D_bias(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention_Rope3D_bias, self).__init__()
        self.config = config
        self.W_Q = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_K = nn.Linear(config.d_embed, config.d_k * config.n_heads, bias=False)
        self.W_V = nn.Linear(config.d_embed, config.d_v * config.n_heads, bias=False)
        self.fc = nn.Linear(config.n_heads * config.d_v, config.d_embed, bias=False)
        self.ln = nn.LayerNorm(config.d_embed)
        self.ia_type_emb=nn.Embedding(402,config.n_heads)

        self.ia_feat_linear = nn.Linear(1, 1, bias=False)

        self.ia_feat_linear = nn.Sequential(
            nn.Linear(6, config.n_heads, bias=False),
            nn.Dropout(0.4),
            nn.ReLU(),
        )
        self.ln_ia_feat = nn.LayerNorm(config.n_heads)

        # pe = torch.zeros(2000, config.d_k//3)
        # position = torch.arange(0, 2000, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, config.d_k//3, 2).float() * (-math.log(10000.0) / config.d_k//3))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)

    def forward(self, input_Q, input_K, input_V, attn_mask,attn_if,interaction_type,interaction_matrix,res_mass_centor,distance_matrix):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        
        # # 创建cosine和sine位置嵌入
        # cos_pos = self.pe[..., 1::2].repeat_interleave(2, dim=-1)
        # sin_pos = self.pe[..., ::2].repeat_interleave(2, dim=-1)

        # # 调整位置嵌入的形状以匹配qw和kw的形状
        # cos_pos = cos_pos.permute(1, 0, 2).expand(1, Q.shape[1], Q.shape[2], Q.shape[3]//3)  # 变为[1, 8, 2000, 16]
        # sin_pos = sin_pos.permute(1, 0, 2).expand(1, Q.shape[1], Q.shape[2], Q.shape[3]//3)  # 变为[1, 8, 2000, 16]

        # Q_x=Q[:,:,:,0:Q.shape[3]//3]
        # Q_y=Q[:,:,:,Q.shape[3]//3:Q.shape[3]//3*2]
        # Q_z=Q[:,:,:,Q.shape[3]//3*2:]

        # # 按照每个batch的第一个坐标维度排序tensorB，并应用到tensorA上
        # sorted_indices = torch.argsort(res_mass_centor[:, :, 0], dim=1)
        # for i in range(Q_x.size(0)):   # 遍历每个batch
        #     Q_x[i] = Q_x[i].index_select(1, sorted_indices[i])
        # q2_x = torch.stack((-Q_x[..., 1::2], Q_x[..., ::2]), dim=-1)
        # q2_x = q2_x.reshape_as(Q_x)
        # Q_x = Q_x * cos_pos + q2_x * sin_pos
        # for i in range(Q_x.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     Q_x[i] = Q_x[i].index_select(1, inverse_sorted_indices)

        # sorted_indices = torch.argsort(res_mass_centor[:, :, 1], dim=1)
        # for i in range(Q_y.size(0)):   # 遍历每个batch
        #     Q_y[i] = Q_y[i].index_select(1, sorted_indices[i])
        # q2_y = torch.stack((-Q_y[..., 1::2], Q_y[..., ::2]), dim=-1)
        # q2_y = q2_y.reshape_as(Q_y)
        # Q_y = Q_y * cos_pos + q2_y * sin_pos
        # for i in range(Q_y.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     Q_y[i] = Q_y[i].index_select(1, inverse_sorted_indices)

        # sorted_indices = torch.argsort(res_mass_centor[:, :, 2], dim=1)
        # for i in range(Q_z.size(0)):   # 遍历每个batch
        #     Q_z[i] = Q_z[i].index_select(1, sorted_indices[i])
        # q2_z = torch.stack((-Q_z[..., 1::2], Q_z[..., ::2]), dim=-1)
        # q2_z = q2_z.reshape_as(Q_z)
        # Q_z = Q_z * cos_pos + q2_z * sin_pos
        # for i in range(Q_z.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     Q_z[i] = Q_z[i].index_select(1, inverse_sorted_indices)

        # Q=torch.cat((Q_x,Q_y,Q_z),dim=3)


        # K_x=K[:,:,:,0:K.shape[3]//3]
        # K_y=K[:,:,:,K.shape[3]//3:K.shape[3]//3*2]
        # K_z=K[:,:,:,K.shape[3]//3*2:]

        # # 按照每个batch的第一个坐标维度排序tensorB，并应用到tensorA上
        # sorted_indices = torch.argsort(res_mass_centor[:, :, 0], dim=1)
        # for i in range(K_x.size(0)):   # 遍历每个batch
        #     K_x[i] = K_x[i].index_select(1, sorted_indices[i])
        # K2_x = torch.stack((-K_x[..., 1::2], K_x[..., ::2]), dim=-1)
        # K2_x = K2_x.reshape_as(K_x)
        # K_x = K_x * cos_pos + K2_x * sin_pos
        # for i in range(K_x.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     K_x[i] = K_x[i].index_select(1, inverse_sorted_indices)

        # sorted_indices = torch.argsort(res_mass_centor[:, :, 1], dim=1)
        # for i in range(K_y.size(0)):   # 遍历每个batch
        #     K_y[i] = K_y[i].index_select(1, sorted_indices[i])
        # K2_y = torch.stack((-K_y[..., 1::2], K_y[..., ::2]), dim=-1)
        # K2_y = K2_y.reshape_as(K_y)
        # K_y = K_y * cos_pos + K2_y * sin_pos
        # for i in range(K_y.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     K_y[i] = K_y[i].index_select(1, inverse_sorted_indices)

        # sorted_indices = torch.argsort(res_mass_centor[:, :, 2], dim=1)
        # for i in range(K_z.size(0)):   # 遍历每个batch
        #     K_z[i] = K_z[i].index_select(1, sorted_indices[i])
        # K2_z = torch.stack((-K_z[..., 1::2], K_z[..., ::2]), dim=-1)
        # K2_z = K2_z.reshape_as(K_z)
        # K_z = K_z * cos_pos + K2_z * sin_pos
        # for i in range(K_z.size(0)):  # 遍历每个batch
        #     _, inverse_sorted_indices = torch.sort(sorted_indices[i])
        #     K_z[i] = K_z[i].index_select(1, inverse_sorted_indices)
        # K=torch.cat((K_x,K_y,K_z),dim=3)
        # ia_type=torch.nn.functional.one_hot(interaction_type.cpu(),num_classes=211).type(torch.float32).cuda().permute(0, 3, 1, 2)
        ia_type=self.ia_type_emb(interaction_type).permute(0, 3, 1, 2)
        # 
        # 那个1到1的线性层，相当于对六个特征求和之后再乘上一个系数
        # ia_feat=self.ia_feat_linear((torch.sum(interaction_matrix,dim=3).float()).reshape(ia_type.shape[0],2000,2000,1)).repeat(1,1,1,self.config.n_heads).permute(0, 3, 1, 2)
        # 那个6到头数的线性层，相当于对六个特征加权求和
        ia_feat=self.ia_feat_linear(interaction_matrix.float()).permute(0, 3, 1, 2)
        
        # ia_type与ia_feat的归一化
        # ia_type=(ia_type-torch.min(ia_type))/(torch.max(ia_type)-torch.min(ia_type))
        # ia_feat=(ia_feat-torch.min(ia_feat))/(torch.max(ia_feat)-torch.min(ia_feat))

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_if = attn_if.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention_bias(self.config)(Q, K, V, attn_mask,attn_if,ia_type,ia_feat,distance_matrix)
        # context, attn = ScaledDotProductAttention_bias(self.config)(Q, K, V, attn_mask,attn_if)

        context = context.transpose(1, 2).reshape(batch_size, -1, self.config.n_heads * self.config.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_embed]
        return self.ln(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_embed, bias=False)
        )
        self.ln = nn.LayerNorm(config.d_embed)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_embed]
        residual = inputs
        output = self.fc(inputs)
        return self.ln(output + residual)  # [batch_size, seq_len, d_embed]

class EncoderLayer_if(nn.Module):
    def __init__(self, config):
        super(EncoderLayer_if, self).__init__()
        self.enc_self_attn = MultiHeadAttention_Rope3D_bias(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, enc_inputs, enc_self_attn_mask,attn_if,interaction_type,interaction_matrix,res_mass_centor,distance_matrix):
        # enc_outputs: [batch_size, pro_len, d_embed], attn: [batch_size, n_heads, pro_len, pro_len]
        # enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask,attn_if,interaction_type,interaction_matrix,res_mass_centor,distance_matrix)  # enc_inputs to same Q,K,V
        
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, pro_len, d_embed]
        return enc_outputs, attn

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_outputs: [batch_size, pro_len, d_embed], attn: [batch_size, n_heads, pro_len, pro_len]
        # enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, pro_len, d_embed]
        return enc_outputs, attn
    

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # self.pos_emb = PositionalEncoding(config.d_embed, max_len=config.pro_len)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layers_en)])

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        # esm用1做pad
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def forward(self, enc_inputs, enc_inputs_true):
        # enc_inputs: [batch_size, pro_len]
        # enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, pro_len, d_embed]
        enc_outputs=enc_inputs
        enc_self_attn_mask = self.get_attn_pad_mask(enc_inputs_true, enc_inputs_true)  # [batch_size, pro_len, pro_len]

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_inputs, enc_self_attn_mask)
        return enc_outputs

class Encoder2(nn.Module):
    def __init__(self, config):
        super(Encoder2, self).__init__()
        # self.pos_emb = PositionalEncoding(config.d_embed, max_len=config.pro_len)
        self.layers = nn.ModuleList([EncoderLayer_if(config) for _ in range(config.n_layers_en)])

    def get_attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        # esm用1做pad
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

    def forward(self, enc_inputs, enc_inputs_true,attn_if,interaction_type,interaction_matrix,res_mass_centor,distance_matrix):
        # enc_inputs: [batch_size, pro_len]
        # enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, pro_len, d_embed]
        enc_self_attn_mask = self.get_attn_pad_mask(enc_inputs_true, enc_inputs_true)  # [batch_size, pro_len, pro_len]
        enc_outputs=enc_inputs

        for layer in self.layers:
            # enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask,attn_if,interaction_type,interaction_matrix,res_mass_centor,distance_matrix)
        return enc_outputs
    


class ESM_linear(nn.Module):
    def __init__(self, config):
        super(ESM_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1280, 64),
            # nn.ReLU(),
            nn.Dropout(0.4),
            # nn.Linear(128, ),
            nn.ReLU()

        )

    def forward(self, pockets):
        return self.fc(pockets)  # (batch_size, d_embed)
    
class ESMIF_linear(nn.Module):
    def __init__(self, config):
        super(ESMIF_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 32),
            nn.Dropout(0.4),
            nn.ReLU()
        )
    def forward(self, pockets):
        return self.fc(pockets)  # (batch_size, d_embed)

class HETATM_linear(nn.Module):
    def __init__(self, config):
        super(HETATM_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(396, 1),
            nn.Dropout(0.4),
            nn.ReLU()
        )
    def forward(self, pockets):
        return self.fc(pockets)  # (batch_size, d_embed)

class MPNN_linear(nn.Module):
    def __init__(self, config):
        super(MPNN_linear, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.Dropout(0.4),
            nn.ReLU()
        )
    def forward(self, pockets):
        return self.fc(pockets)  # (batch_size, d_embed)


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        # self.esm2, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.layer_norm_esm = nn.LayerNorm([config.pro_len,1280],elementwise_affine=False)
        self.layer_norm_esmif = nn.LayerNorm([config.pro_len,512],elementwise_affine=False)
        # self.layer_norm_hetatm = nn.LayerNorm([config.pro_len,396])
        self.layer_norm_hetatm = nn.LayerNorm([config.pro_len,396],elementwise_affine=False)
        # self.layer_norm_esm_pretrain = nn.LayerNorm([config.pro_len,1280],elementwise_affine=False)
        # self.layer_norm_mpnn = nn.LayerNorm([config.pro_len,128],elementwise_affine=False)
        # self.mamba_model = create_sequence_to_vector_model(config.pro_vocab_size)
        self.compress_esm = ESM_linear(config)
        self.compress_esmif = ESMIF_linear(config)
        self.compress_hetatm = HETATM_linear(config)
        # self.compress_esm_pretrain = ESM_linear(config)
        # self.compress_mpnn = MPNN_linear(config)

        # self.encoder=Encoder(config)
        self.encoder2=Encoder2(config)

        # self.projection =nn.Sequential(
        #     nn.Linear(97, 64),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(64, 33),
        #     nn.Dropout(0.3)
        # )
        self.projection =nn.Linear(97, 33)


        self.projection2=nn.Linear(config.pro_len,1)
        self.config = config
    def forward(self, enc_tokens, coor_features,hetatm_features, interface_atoms,\
                interaction_type,interaction_matrix,res_mass_centor,seqs,protein_names,chain_id_res,mpnn_features,\
                    masked_token,mask_padG):


        # for name, param in self.esm2.named_parameters():
        #     if 'layers.32' not in name:
        #         param.requires_grad = False
        self.esm2=self.esm2.eval()
        for name, param in self.esm2.named_parameters():
            param.requires_grad = False

        # print(masked_token.shape)
        # print(mask_padG.shape)
        with torch.no_grad():
            enc_outputs1 = self.esm2(masked_token.type(torch.int32), repr_layers=[33], return_contacts=False)["representations"][33]
        # print(enc_outputs1.shape)

            enc_outputs1 = enc_outputs1[:,mask_padG.squeeze(0),:]
            # print(enc_outputs1.shape)
            enc_outputs1=F.pad(enc_outputs1,(0,0,0,2000-enc_outputs1.shape[1]))


            # print(protein_names)


        enc_outputs1 = self.layer_norm_esm(enc_outputs1)

        enc_outputs1 = self.compress_esm(enc_outputs1)

        # seq_features = self.layer_norm_esm(seq_features)
        coor_features = self.layer_norm_esmif(coor_features)
        # mpnn_features = self.layer_norm_mpnn(mpnn_features)


        hetatm_features =  self.layer_norm_hetatm(hetatm_features)

        # seq_features_pretrain = self.layer_norm_esm_pretrain(seq_features_pretrain)

        # enc_outputs_mamba=self.mamba_model(enc_tokens)
        
        # enc_outputs1 = self.compress_esm(seq_features)
        enc_outputs2 = self.compress_esmif(coor_features)

        # enc_outputs3 =  self.compress_mpnn(mpnn_features)

        enc_outputs4 =  self.compress_hetatm(hetatm_features)
        # enc_outputs5 = self.compress_esm_pretrain(seq_features_pretrain)

        enc_outputs=torch.cat((enc_outputs1,enc_outputs2,enc_outputs4), dim=2)
        # print(res_mass_centor.shape)

        # distance_matrix = torch.cdist(res_mass_centor, res_mass_centor)
        # for batch in range(len(chain_id_res)):
        #     distance_matrix[batch,len(chain_id_res[batch]):,:]=0
        #     distance_matrix[batch,:,len(chain_id_res[batch]):]=0
        # distance_matrix=(-distance_matrix/torch.max(distance_matrix)).add(1)
        # distance_matrix[distance_matrix >= 1] = 0

        # enc_outputs_enc=self.encoder2(enc_outputs,enc_tokens,interface_atoms,interaction_type,interaction_matrix,res_mass_centor,distance_matrix)

        # # # enc_outputs=self.encoder(enc_outputs,enc_tokens)
        # # # enc_outputs = self.projection(enc_outputs).reshape(enc_outputs.shape[0],-1)   # dec_logits: [batch_size, tgt_len, vocab_size]
        # # # enc_outputs=torch.cat((enc_outputs,enc_outputs_enc), dim=2)
        
        # # # enc_outputs = self.projection2(enc_outputs.permute(0,2,1)).reshape(enc_outputs.shape[0],-1)
        
        # enc_outputs=torch.cat((enc_outputs,enc_outputs_enc), dim=2)


        # enc_outputs = torch.mean(enc_outputs,dim=1)

        # # enc_outputs_f1=self.encoder(enc_outputs,enc_tokens)
        # enc_outputs_f=self.encoder2(enc_outputs,enc_tokens,interface_atoms)
        # # enc_outputs_f=torch.cat((enc_outputs_f1, enc_outputs_f2), dim=2)
        # enc_outputs = self.projection(enc_outputs_f).reshape(enc_outputs_f.shape[0],-1)   # dec_logits: [batch_size, tgt_len, vocab_size]
        
        # dec_logits = self.projection(enc_outputs)  # dec_logits: [batch_size, tgt_len, vocab_size]
        
        dec_logits = self.projection(enc_outputs)



        return dec_logits

