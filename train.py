#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import utils

from models.model import LogReg, LinReg
from trainer import Trainer

def f(x:torch.Tensor, a:float, b:int) -> torch.Tensor:
    eps = torch.normal(0, 1, size=(len(x),))
    y = a*x + b + eps
    return y

def phi(X, a, b):
    return torch.tensor([[b, x, np.cos(a*x), np.sin(a*x), -np.cos(a*x), -np.sin(a*x)] for x in X])

def run():
    N = 300
    X = torch.arange(N)
    a = torch.rand(1)
    b = 1
    y = f(x, a, b)
    fraction = 2/3
    X = phi(X, a, b)
    X_train, X_val, y_val, y_train = utils.train_val_split(X, y, fraction)
    train_dataset = utils.gen_dataset(X_train, y_train)
    val_dataset = utils.gen_dataset(X_val, y_val)
    train_batches = utils.gen_batches(train_dataset, batch_size=batch_size, shuffle=True)
    val_batches = utils.gen_batches(val_dataset, batch_size=batch_size, shuffle=True)

    for k, criterion in enumerate(criteria):
        trainer = Trainer(
                         lr=lr,
                         batch_size=batch_size,
                         max_iter=max_iter,
                         steps=steps,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device,
                         window_size=window_size,
        )

        trainer.fit(model, )
