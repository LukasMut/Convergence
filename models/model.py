#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ['LogReg', 'MLP', 'VReg']

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogReg(nn.Module):

    def __init__(self, in_size:int):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_features=in_size, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        y = self.sigmoid(self.fc(x))
        return y

class LinReg(nn.Module):

    def __init__(self, in_size:int, out_size:int):
        super(LinReg, self).__init__()
        self.fc = nn.Linear(in_features=in_size, out_features=out_size, bias=True)

    def forward(self, x:torch.Tensor):
        return self.fc(x)

class MLP(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(in_features=in_size, out_features=hidden_size, bias=True)
        self.fc_2 = nn.Linear(in_features=hidden_size, out_features=out_size, bias=True)

    def forward(self, x:torch.Tensor):
        z = self.fc_1(x)
        y = self.fc_2(z)
        return y

class VReg(nn.Module):

    def __init__(self, in_size:int, out_size:int):
        super(VReg, self).__init__()
        self.encoder_mu = nn.Linear(in_features=in_size, out_features=out_size, bias=False)
        self.encoder_logvar = nn.Linear(in_features=in_size, out_features=out_size, bias=False)

    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        sigma = logvar.mul(0.5).exp()
        eps = sigma.data.new(sigma.size()).normal_()
        return eps.mul(sigma).add_(mu)

    def forward(self, x:torch.Tensor):
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        W_mu = self.encoder_mu.weight.data.T
        W_logvar = self.encoder_mu.weight.data.T
        W_sampled = self.reparameterize(W_mu, W_logvar)
        z = torch.mm(x, W_sampled)
        return z, W_mu, W_logvar, W_sampled
