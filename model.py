import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from utils import seq2sen

class SAT(nn.Module):
    def __init__(self, args):
        super(SAT, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features.children())[:-1])
        self.decoder = LSTM(args)
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, inputs, target_length, answers=None):
        # inputs: (batch x channels x height x width)
        # features: (batch x num_regions x embedding_dim)
        # outputs: (batch x target_len x embedding_dim)
        # attns: (batch x num_regions)
        features = self.encoder(inputs)
        shape = features.shape
        features = features.view(shape[0], shape[1], -1)
        features = features.transpose(1, 2)
#         print(features.shape)
        outputs, attns = self.decoder(features, target_length, answers=answers)
        return outputs, attns
    def infer(self, inputs):
        features = self.encoder(inputs)
        shape = features.shape
        features = features.view(shape[0], shape[1], -1)
        features = features.transpose(1, 2)
        outputs, attns = self.decoder.infer(features)
        return outputs, attns

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.layers = nn.LSTMCell(args.embedding_dim + args.word_dim, args.hidden_dim)
        self.layer_h0 = nn.Linear(args.embedding_dim, args.hidden_dim)
        self.layer_c0 = nn.Linear(args.embedding_dim, args.hidden_dim)
        self.attention = Attention(args)
        self.Lo = nn.Linear(args.word_dim, args.vocab_size)
        self.Lh = nn.Linear(args.hidden_dim, args.word_dim)
        self.Lz = nn.Linear(args.embedding_dim, args.word_dim)
        self.word_embedding = nn.Embedding(args.vocab_size, args.word_dim)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, inputs, target_len, answers=None):
        # inputs: (batch x num_regions x embedding_dim)
        # answers: (batch x target_len)
        # outputs: (batch x target_len x vocab_size)
        # attns: (batch x target_len x num_regions)
        h = nn.Tanh()(self.layer_h0(inputs.mean(dim=1)))
        c = nn.Tanh()(self.layer_c0(inputs.mean(dim=1)))
        outputs = torch.Tensor(0).to(self.args.device)
        attns = torch.Tensor(0).to(self.args.device)
        if answers is None:
            last_words = torch.Tensor(inputs.shape[0]).long().to(self.args.device)
            last_words[:] = self.args.sos_idx 
        for i in range(target_len):
            # z: (batch x embedding_dim)
            # word_embedding: (batch x word_dim)
            z, attn = self.attention(inputs, h)
            attns = torch.cat((attns, attn.unsqueeze(1)), dim=1)
            if answers is not None:
                word_embedding = self.word_embedding(answers[:, i])
            else:
                word_embedding = self.word_embedding(last_words)
            h, c = self.layers(torch.cat((z, word_embedding), dim=1), (h, c))
            outputs = torch.cat((outputs, self.Lo(word_embedding + self.Lh(self.dropout(h)) + self.Lz(z)).unsqueeze(1)), dim=1)
            # outputs = torch.cat((outputs, self.Lh(self.dropout(h))), dim=1)
            if answers is None:
                last_words = outputs[:, -1].argmax(dim=1)
        return outputs, attns

    def infer(self, inputs):
        # inputs: (batch x num_regions x embedding_dim)
        # outputs: (batch x target_len)
        h = self.layer_h0(inputs.mean(dim=1))
        c = self.layer_c0(inputs.mean(dim=1))
        batch = inputs.shape[0]
        outputs = torch.Tensor(batch, 1).long().to(self.args.device)
        outputs[:, :] = self.args.sos_idx
        is_terminated = torch.zeros(batch).bool().to(self.args.device)
        attns = torch.Tensor(0).to(self.args.device)
        for i in range(self.args.max_target_length):
            # z: (batch x embedding_dim)
            # word_embedding: (batch x word_dim)
            z, attn = self.attention(inputs, h)
            attns = torch.cat((attns, attn.unsqueeze(1)), dim=1)
            word_embedding = self.word_embedding(outputs[:, -1])
            h, c = self.layers(torch.cat((z, word_embedding), dim=1), (h, c))
            # preds = self.Lh(self.dropout(h))
            preds = self.Lo(word_embedding + self.Lh(h) + self.Lz(z))
            new_words = preds.argmax(dim=1).masked_fill_(is_terminated, self.args.pad_idx)
            is_terminated = is_terminated.logical_or(new_words == self.args.eos_idx)
            outputs = torch.cat((outputs, new_words.unsqueeze(1)), dim=1)          
            if is_terminated.sum() == len(is_terminated):
                break
        outputs = torch.cat((outputs, torch.ones_like(is_terminated).unsqueeze(1)), dim=1)
        outputs[:, -1] = self.args.eos_idx
        return outputs, attns

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.Lh = nn.Linear(args.hidden_dim, args.attn_dim)
        self.Lf = nn.Linear(args.embedding_dim, args.attn_dim)
        self.Le = nn.Linear(args.attn_dim, 1)
    def forward(self, features, h):
        # features: (batch x num_regions x embedding_dim)
        # h: (batch x hidden_dim)
        # e: (batch x num_regions)
        # attn: (batch x num_regions)
        # context: (batch x embedding_dim)
        e = self.Le(nn.Tanh()(self.Lf(features) + self.Lh(h).unsqueeze(1))).squeeze(2)
        attn = nn.Softmax(dim=1)(e)
        context = (attn.unsqueeze(2) * features).sum(dim=1)
        return context, attn
        
