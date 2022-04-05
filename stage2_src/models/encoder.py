import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
         
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        # self.img_fn= nn.Sequential(
        #     nn.Linear(2048,1024),
        #     nn.Linear(1024,768))
    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
        
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
    
class ImageEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(ImageEncoderLayer, self).__init__()

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, key, value, query, mask):
        if (iter != 0):
            key = self.layer_norm(key)
            value = self.layer_norm(value)
            query = self.layer_norm(query)
        else:
            pass

        mask = mask.unsqueeze(1)
         
        context = self.context_attn(key, value, query,
                                 mask=mask)
        out = self.dropout(context) + query
        return self.feed_forward(out)
   
class ExtImageTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtImageTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.num_image_layer=1
        self.transformer_image_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        #768
            
        self.context_inter=nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
         
       
    def forward(self, top_vecs, mask,ic_top_vecs,mask_ic):
        """ See :obj:`EncoderBase.forward()`"""
        
#         im_f = self.img_fn(image_data).unsqueeze(1).repeat(1,top_vecs.shape[1],1)
#         print(image_data.shape)
#         print(top_vecs.shape)
#         print(mask.shape)
         
#         print(type(mask),type(image_mask))
#         print(image_mask)
#         print(mask)
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#         x = torch.cat([x,im_f],2)
        x = self.layer_norm(x)
        y = ic_top_vecs* mask_ic[:, :, None].float()
        for i in range(self.num_image_layer):
            y = self.transformer_image_inter[i](i,y,y, ~mask_ic)
        
        y = self.layer_norm(y)
        for i in range(self.num_image_layer):
            x = self.context_inter[i](i,y,y,x, ~mask_ic)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
    
class ExtImageTransformerEncoder2(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtImageTransformerEncoder2, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.num_image_layer=2
        self.transformer_image_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        #768
            
        self.context_inter=nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.context_inter2 = nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
       
    def forward(self, top_vecs, mask,ic_top_vecs,mask_ic):
        """ See :obj:`EncoderBase.forward()`"""
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#         x = torch.cat([x,im_f],2)
        x = self.layer_norm(x)
        
        y = ic_top_vecs* mask_ic[:, :, None].float()
        for i in range(self.num_image_layer):
            y = self.transformer_image_inter[i](i,y,y, ~mask_ic)
        
        y = self.layer_norm(y)
        res_y = y
        for i in range(self.num_image_layer):
            res_y = self.context_inter[i](i,x,x,res_y, ~mask)
        res_y = self.layer_norm(res_y)
        y = y+res_y
       
        res_x =x 
        for i in range(self.num_image_layer):
            res_x = self.context_inter2[i](i,y,y,res_x, ~mask_ic)
        res_x = self.layer_norm(res_x)
         
        x = x+res_x
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
class ExtImageTransformerEncoder3(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtImageTransformerEncoder3, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.num_image_layer=2
        self.transformer_image_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        #768
            
        self.context_inter=nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.context_inter2 = nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.ic_gate = nn.Sequential(
           nn.Linear(d_model,1),
           nn.Sigmoid()
           )
        self.src_gate=nn.Sequential(
           nn.Linear(d_model,1),
           nn.Sigmoid()
           )
    def forward(self, top_vecs, mask,ic_top_vecs,mask_ic):
        """ See :obj:`EncoderBase.forward()`"""
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#         x = torch.cat([x,im_f],2)
        x = self.layer_norm(x)
        
        y = ic_top_vecs* mask_ic[:, :, None].float()
        for i in range(self.num_image_layer):
            y = self.transformer_image_inter[i](i,y,y, ~mask_ic)
        
        y = self.layer_norm(y)
        res_y = y
        for i in range(self.num_image_layer):
            res_y = self.context_inter[i](i,x,x,res_y, ~mask)
        y = y*self.ic_gate(res_y)
        y = self.layer_norm(y)
        res_x =x 
        for i in range(self.num_image_layer):
            res_x = self.context_inter2[i](i,y,y,res_x, ~mask_ic)
        res_x = res_x*self.src_gate(res_x)
        x = x+res_x
        x=self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores