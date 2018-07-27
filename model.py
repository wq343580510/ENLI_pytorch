# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy
import init
from init import *


# encoder
# attention
# projection   relu
# decoder
# ff_layer_1   tanh
# ff_layer_output  linear


class ENLI_Model(nn.Module):
    def __init__(self,config,worddicts = None,load_pretrain = True):
        super(ENLI_Model, self).__init__()

        self.word_dim = config.get('word_dim')
        self.drop_rate = config.get('dropout')
        self.dropout_layer = nn.Dropout(self.drop_rate)
        self.word_embs = nn.Embedding(config['n_words'],self.word_dim,padding_idx=0)

        self.encoder = nn.LSTM(config.get('word_dim'), config.get('hidden_units'),bidirectional=True)
        self.decoder = nn.LSTM(config.get('hidden_units'), config.get('hidden_units'),bidirectional=True)
        self.projection = nn.Sequential(
            nn.Linear(8 * config.get('hidden_units'), config.get('hidden_units')),
            nn.ReLU()
        )
        self.ff_layer_1 = nn.Sequential(
            nn.Linear(8 * config.get('hidden_units'), config.get('hidden_units')),
            nn.Tanh()
        )
        self.ff_layer_output = nn.Sequential(
            nn.Linear(config.get('hidden_units'), 4),
            nn.Softmax(dim=-1)
        )
        if load_pretrain:
            self.init_pretrain(config['embeddings'],config['n_words'],worddicts)


    def attention_layer(self,ctx1,ctx2,x1_mask,x2_mask):
        # ctx1 : batch_size x len1 x dim
        # ctx2 : batch_size x len2 x dim
        # x1_mask : batch_size x len1
        # x2_mask : batch_size x len2

        #32 x 13 x 25
        weight_matrix = torch.matmul(ctx1, ctx2.transpose(1, 2))

        #32 x 13 x 25
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0])
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0])
        #32 x 13 x 25    32 x 13 x 1
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, :, None]
        #32 x 13 x 25    32 x 1 x 25
        weight_matrix_2 = weight_matrix_2 * x2_mask[:, None, :]

        # alpha shape 32 x 13 x 25 (13归一化)
        # beta shape 32 x 13 x 25 (25归一化)
        alpha = weight_matrix_1 / weight_matrix_1.sum(1, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(2, keepdim=True)

        # ctx1 32 x 13 x 1 x dim     alpha 32 x 13 x 25 x 1
        ctx2_ = (torch.unsqueeze(ctx1,2) * torch.unsqueeze(alpha,3)).sum(1)
        # ctx2 32 x 1 x 25 x dim   beta 32 x 13 x 25 x1
        ctx1_ = (torch.unsqueeze(ctx2,1) * torch.unsqueeze(beta,3)).sum(2)

        inp1 = torch.cat([ctx1, ctx1_, ctx1 * ctx1_, ctx1 - ctx1_], dim=2)
        inp2 = torch.cat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], dim=2)

        return inp1,inp2,alpha,beta

    def pooling_layer(self,ctx1,mask1,ctx2,mask2):
        logit1 = ctx1.sum(1) / mask1.sum(1,keepdim=True)
        logit2 = ctx1.max(1)[0]
        logit3 = ctx2.sum(1) / mask2.sum(1,keepdim=True)
        logit4 = ctx2.max(1)[0]
        return torch.cat([logit1,logit2,logit3,logit4],dim=1)

    def init_pretrain(self,filename,n_words,worddicts):
        count = 0
        print('load pretrain')
        for line in open(filename, 'r',encoding='utf8').readlines():
            tmp = line.split()
            word = tmp[0]
            vector = tmp[1:]
            if word in worddicts and worddicts[word] < n_words:
                count += 1
                self.word_embs.weight.data[worddicts[word]].copy_(torch.from_numpy(np.array(vector,dtype='float32')))
        print('I found {} words in the embedding file'.format(count))


    def init_params(self):
        init.bilstm_init(self.encoder)
        init.bilstm_init(self.decoder)
        init.embeddings_init(self.word_embs)
        init.fflayer_init(self.ff_layer_1[0], False)
        init.fflayer_init(self.ff_layer_output[0], False)
        init.fflayer_init(self.projection[0], False)



    def forward(self,x1,x1_mask,x2,x2_mask,ret_att = False):
        #look up word embedding  32 x 13 x 200  -> 13 x 32 x 200
        x1_emb = self.dropout_layer(self.word_embs(x1).transpose(0,1))
        x2_emb = self.dropout_layer(self.word_embs(x2).transpose(0,1))

        # encoder layer
        enc_1 = (self.encoder(x1_emb)[0]).transpose(0,1) * x1_mask[:, :, None]
        enc_2 = (self.encoder(x2_emb)[0]).transpose(0,1) * x2_mask[:, :, None]

        # attention layer
        att_1,att_2,alpha,beta = self.attention_layer(enc_1,enc_2,x1_mask,x2_mask)
        if ret_att:
            return alpha,beta
        # projection layer
        att_1 = self.dropout_layer(self.projection(att_1.transpose(0,1)))
        att_2 = self.dropout_layer(self.projection(att_2.transpose(0,1)))

        # decoder layer
        dec_1 = (self.decoder(att_1)[0]).transpose(0,1) * x1_mask[:, :, None]
        dec_2 = (self.decoder(att_2)[0]).transpose(0,1) * x2_mask[:, :, None]

        # projector
        logit = self.dropout_layer(self.pooling_layer(dec_1,x1_mask,dec_2,x2_mask))

        logit = self.dropout_layer(self.ff_layer_1(logit))
        logit = self.ff_layer_output(logit)

        return logit

