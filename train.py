#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import random
import os
import torch
import utils

import copy
import numpy as np

from collections import defaultdict
from models.model import LogReg, LinReg
from trainer import Trainer
from typing import List, Tuple

def parseargs():
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('--optimizer', type=str,
        choices=['SGD', 'RMSprop', 'Adam', 'AdamW'])
    aa('--learning_rate', type=float, default=0.001,
        help='step size multiplied by batch-averaged gradients during each iteration')
    aa('--batch_size', metavar='B', type=int, default=10)
    aa('--max_epochs', metavar='T', type=int, default=100,
        help='maximum number of epochs to optimize VSPoSE for')
    aa('--num_samples', metavar='N', type=int,
        help='number of samples to be drawn')
    aa('--criteria', type=str, nargs='+',
        help='list of different convergence criteria to be tested')
    aa('--window_size', type=int, default=50,
        help='window size to be used for checking convergence criterion with linear regression')
    aa('--steps', type=int,
        help='perform validation and save model parameters every <steps> epochs')
    aa('--results_dir', type=str,
        help='path/to/results')
    aa('--device', type=str, default='cpu',
        choices=['cpu', 'cuda'])
    aa('--rnd_seed', type=int, default=42,
        help='random seed for reproducibility')
    args = parser.parse_args()
    return args

def f(x:torch.Tensor, a:float, b:int) -> torch.Tensor:
    eps = torch.normal(0, 1, size=(len(x),))
    y = a*x + b + eps
    return y

def phi(X, a, b):
    return torch.tensor([[b, x, np.cos(a*x), np.sin(a*x), -np.cos(a*x), -np.sin(a*x)] for x in X])

def copy_model(model, criteria:List[str]) -> list:
    return [copy.deepcopy(model) for _ in range(len(criteria))]

def run(
        optimizer:str,
        lr:float,
        batch_size:int,
        max_epochs:int,
        num_samples:int,
        criteria:List[str],
        window_size:int,
        steps:int,
        device:torch.device,
        results_dir:str,
) -> None:
    X = torch.linspace(1, 2, num_samples)
    a = torch.rand(1)
    b = 1
    y = f(X, a, b)
    fraction = 2/3
    X = phi(X, a, b)
    X_train, X_val, y_val, y_train = utils.train_val_split(X, y, fraction)
    train_batches = utils.gen_batches(utils.gen_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_batches = utils.gen_batches(utils.gen_dataset(X_val, y_val), batch_size=batch_size, shuffle=True)
    results = defaultdict(dict)
    model = LinReg(in_size=X_train.shape[1], out_size=1)
    models = copy_model(model, criteria)
    for k, criterion in enumerate(criteria):
        model = models[k]
        trainer = Trainer(
                         lr=lr,
                         batch_size=batch_size,
                         max_epochs=max_epochs,
                         steps=steps,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device,
                         results_dir=results_dir,
                         window_size=window_size,
        )
        steps, train_losses, val_losses = trainer.fit(model=model, train_batches=train_batches, val_batches=val_batches, verbose=True)
        print(f'Finished optimization for {criterion} criterion after {steps} steps\n')

        results[criterion]['stopping'] = steps
        results[criterion]['train_losses'] = train_losses
        results[criterion]['val_losses'] = val_losses

    _save_results(trainer, results)

def _save_results(trainer:object, results:dict) -> None:
    with open(os.path.join(trainer.PATH, 'results.txt'), 'wb') as f:
        f.write(pickle.dumps(results))

if __name__ == '__main__':
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    device = torch.device(args.device)
    run(
        optimizer=args.optimizer,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_samples=args.num_samples,
        criteria=args.criteria,
        window_size=args.window_size,
        steps=args.steps,
        device=device,
        results_dir=args.results_dir,
        )
