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
         
        context,attn = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out),attn


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
        self.img_fn= nn.Sequential(
            nn.Linear(2048,1024),
            nn.Linear(1024,768))
    def forward(self, top_vecs, mask ):
        """ See :obj:`EncoderBase.forward()`"""
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        attn_list= []
        for i in range(self.num_inter_layers):
            x,attn = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
            attn_list.append(attn)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores,attn_list
    
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
         
        context,attn = self.context_attn(key, value, query,
                                 mask=mask)
        out = self.dropout(context) + query
        return self.feed_forward(out),attn
   
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
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        #768
            
        self.context_inter=nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.img_fn= nn.Sequential(
            nn.Linear(2048,1024),
            nn.Linear(1024,768))
       
    def forward(self, top_vecs, mask,image_data,image_mask):
        """ See :obj:`EncoderBase.forward()`"""
        
#         im_f = self.img_fn(image_data).unsqueeze(1).repeat(1,top_vecs.shape[1],1)
#         print(image_data.shape)
#         print(top_vecs.shape)
#         print(mask.shape)
        image_data = self.img_fn(image_data).float()
        # image_mask = torch.Tensor([True]*image_data.shape[1]).unsqueeze(0).repeat(image_data.shape[0],1).cuda().bool()
#         print(type(mask),type(image_mask))
#         print(image_mask)
#         print(mask)
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x,_ = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#         x = torch.cat([x,im_f],2)
        x = self.layer_norm1(x)
    
        for i in range(self.num_image_layer):
            image_data,_ = self.transformer_image_inter[i](i,image_data,image_data, ~image_mask)
        image_data = self.layer_norm2(image_data)
        attn_list =[]
        for i in range(self.num_image_layer):
            x,attn = self.context_inter[i](i,image_data,image_data,x, ~image_mask)
            attn_list.append(attn)
        x = self.layer_norm3(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores,attn_list
    
class ExtImageTransformerEncoder2(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtImageTransformerEncoder2, self).__init__()
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
        self.img_fn= nn.Sequential(
            nn.Linear(2048,1024),
            nn.Linear(1024,768))
       
    def forward(self, top_vecs, mask,image_data):
        """ See :obj:`EncoderBase.forward()`"""
        
#         im_f = self.img_fn(image_data).unsqueeze(1).repeat(1,top_vecs.shape[1],1)
#         print(image_data.shape)
#         print(top_vecs.shape)
#         print(mask.shape)
        image_data = self.img_fn(image_data).float()
        image_mask = torch.Tensor([True]*image_data.shape[1]).unsqueeze(0).repeat(image_data.shape[0],1).cuda().bool()
#         print(type(mask),type(image_mask))
#         print(image_mask)
#         print(mask)
        
        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x,_ = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
#         x = torch.cat([x,im_f],2)
        x = self.layer_norm(x)
        
        for i in range(self.num_image_layer):
            image_data,_ = self.transformer_image_inter[i](i,image_data,image_data, ~image_mask)

        image_data = self.layer_norm(image_data)
        attn_list=[]
        for i in range(self.num_image_layer):
            print(image_data.shape, x.shape)
            x,attn = self.context_inter[i](i,image_data,image_data,x, ~image_mask)
            attn_list.append(attn)
        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores,attn_list


class ExtImageTransformerEncoder_image_score(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtImageTransformerEncoder_image_score, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.num_image_layer = 1
        self.transformer_image_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm4 = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
        self.sigmoid = nn.Sigmoid()
        # 768

        self.context_inter = nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.image_inter = nn.ModuleList(
            [ImageEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(self.num_image_layer)])
        self.img_fn = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Linear(1024,d_model))
        self.image_wo = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.Linear(d_model, 1, bias=True))
    def forward(self, top_vecs, mask, image_data,image_mask):
        """ See :obj:`EncoderBase.forward()`"""

        #         im_f = self.img_fn(image_data).unsqueeze(1).repeat(1,top_vecs.shape[1],1)
        #         print(image_data.shape)
        #         print(top_vecs.shape)
        #         print(mask.shape)
        image_data = self.img_fn(image_data).float()

        #         print(type(mask),type(image_mask))
        #         print(image_mask)
        #         print(mask)

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb
        for i in range(self.num_inter_layers):
            x, _ = self.transformer_inter[i](i, x, x, ~mask)  # all_sents * max_tokens * dim
        #         x = torch.cat([x,im_f],2)
        x = self.layer_norm1(x)

        for i in range(self.num_image_layer):
            image_data, _ = self.transformer_image_inter[i](i, image_data, image_data, ~image_mask)
        image_data = self.layer_norm2(image_data)

        attn_list = []
        new_x = x
        for i in range(self.num_image_layer):
            new_x, attn = self.context_inter[i](i, image_data, image_data, new_x, ~image_mask)
            attn_list.append(attn)
        new_x = new_x+x
        new_x = self.layer_norm3(new_x)

        new_image_data = image_data
        for i in range(self.num_image_layer):
            new_image_data, _ = self.image_inter[i](i, x, x, new_image_data, ~mask)
        new_image_data = new_image_data+image_data
        new_image_data= self.layer_norm4(new_image_data)

        sent_scores = self.sigmoid(self.wo(new_x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()
        image_scores = self.sigmoid(self.image_wo(new_image_data))
        image_scores = image_scores.squeeze(-1) * image_mask.float()
        return sent_scores, attn_list,image_scores