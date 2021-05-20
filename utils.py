#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def register_hooks(model):
    """register a backward hook to store per-sample gradients"""
    for m in model.modules():
        m.register_forward_hook(collect_acts)
        m.register_backward_hook(collect_grads)
    return model

def collect_acts(layer, input, output):
    """store per-sample gradients (weights are shared between encoders)"""
    setattr(layer, 'inputs', input[0].detach())

def collect_grads(layer, input, output):
    """store per-sample gradients (weights are shared between encoders)"""
    global gradients
    gradients = []
    gradients.append(output[0].detach())

def clear_backprops(model: nn.Module) -> None:
    """remove gradient information in every layer"""
    for m in model.modules():
        if hasattr(m, 'gradients'):
            del m.gradients

def compute_sample_grads(model) -> None:
    for n, m in model.named_modules():
        if n.startswith('encoder'):
            A = m.inputs
            M = A.shape[0]
            B = gradients[0] * M
            setattr(m.weight, 'sample_gradients', torch.einsum('ni,nj->nij', B, A))

def estimate_grad_var(sample_gradients, average_gradients):
    return (sample_gradients - average_gradients[None, ...]).pow(2).sum(dim=0)/(sample_gradients.shape[0]-1)

def criterion(avg_grad:torch.Tensor, var_estimator:torch.Tensor, batch_size:int):
    D = avg_grad.shape[0] * avg_grad.shape[1]
    return 1 - (batch_size/D)*((avg_grad.pow(2)/var_estimator).sum())

def get_means_and_vars(model) -> Tuple[torch.Tensor, torch.Tensor]:
    avg_gradients, var_estimators = zip(*[(p.grad, estimate_grad_var(p.sample_gradients, p.grad)) for p in model.parameters()])
    avg_grad = torch.cat(avg_gradients, dim=0)
    var_estimator = torch.cat(var_estimators, dim=0)
    return avg_grad, var_estimator