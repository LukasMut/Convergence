#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import utils
import math

import numpy as np
import torch.nn.functional as F

from torch.optim import SGD, RMSprop, Adam, AdamW
from scipy.stats import linregress
from typing import Iterator, List, Tuple

optimizers = ['SGD', 'RMSprop', 'Adam', 'AdamW']
eb_criteria = ['gradients', 'hessian']

def mse(y:torch.Tensor, y_hat:torch.Tensor) -> torch.Tensor:
    return (y - y_hat).mul(0.5).pow(2).mean()

def pdf(sample:torch.Tensor, mu:torch.Tensor, sigma:torch.Tensor):
    return torch.exp(-((sample - mu) ** 2) / (2 * sigma.pow(2))) / sigma * math.sqrt(2 * math.pi)

class Trainer(object):

    def __init__(
                 self,
                 lr:float,
                 batch_size:int,
                 max_epochs:int,
                 steps:int,
                 criterion:str,
                 optimizer:str,
                 device:torch.device,
                 results_dir:str,
                 window_size:int=None,
        ):
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.steps = steps
        self.criterion = criterion
        self.device = device
        self.results_dir = results_dir
        self.window_size = window_size

        assert optimizer in optimizers, f'Optimizer must be one of {optimizers}\n'

        self.optimizer = optimizer

        self.PATH = os.path.join(self.results_dir, self.optimizer)
        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

    def get_optim(self, model):
        if self.optimizer == 'SGD':
            optim = SGD(model.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            optim = RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        elif self.optimizer == 'Adam':
            optim = Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            optim = AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        return optim

    def validation(self, model, val_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]]):
        model.eval()
        val_losses = torch.zeros(len(val_batches))
        with torch.no_grad():
            for i, batch in enumerate(val_batches):
                batch = tuple(t.to(self.device) for t in batch)
                X, y = batch
                y_hat = model(X)
                loss = mse(y, y_hat)
                val_losses[i] += loss.item()
        avg_val_loss = torch.mean(val_losses).item()
        return avg_val_loss

    def early_stopping(
                       self,
                       model,
                       val_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]],
                       val_losses:List[float],
                       train_losses:List[float]=None,
                       steps:int=None,
                       ) -> Tuple[bool, float]:
        stop_trainig = False
        if self.criterion == 'gradients':
            assert self.batch_size, '\nWhen evaluating an evidence-based stopping criterion, batch size must be provided\n'
            utils.compute_sample_grads(model)
            avg_grad, var_estimator = utils.get_means_and_vars(model)
            eb_criterion = utils.eb_criterion(avg_grad, var_estimator, self.batch_size)
            if eb_criterion > 0:
                stop_trainig = True

        elif self.criterion == 'hessian':
            raise NotImplementedError

        elif self.criterion == 'validation':
            current_val_loss = self.validation(model, val_batches)
            val_losses.append(current_val_loss)
            assert self.window_size, '\nWhen evaluating the mse on a validation set, window size parameter is required\n'
            if (steps + 1) > self.window_size:
                lmres = linregress(range(self.window_size), val_losses[(steps + 1 - self.window_size):(steps + 2)])
                if (lmres.slope > 0) or (lmres.pvalue > .1):
                    stop_trainig = True

        elif self.criterion == 'training':
            assert self.window_size, '\nWhen evaluating the mse on the train set, window size parameter is required\n'
            if (steps + 1) > self.window_size:
                lmres = linregress(range(self.window_size), train_losses[(steps + 1 - self.window_size):(steps + 2)])
                if (lmres.slope > 0) or (lmres.pvalue > .1):
                    stop_trainig = True

        if self.criterion != 'validation':
            current_val_loss = self.validation(model, val_batches)
            val_losses.append(current_val_loss)
        return stop_trainig, val_losses

    def _save_params(self, model, optim, steps:int) -> None:
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    }, os.path.join(self.PATH, f'model_steps{steps+1:04d}.tar'))
    def fit(
            self,
            model,
            train_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]],
            val_batches=None,
            verbose:bool=True,
            ):
        model.to(self.device)
        optim = self.get_optim(model)
        if self.criterion in eb_criteria:
            model = utils.register_hooks(model)
        train_losses, val_losses = [], []
        steps = 0
        stopping = False
        for epoch in range(self.max_epochs):
            for i, batch in enumerate(train_batches):
                model.train()
                optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                X, y = batch
                y_hat = model(X)
                loss = mse(y, y_hat)
                loss.backward()
                train_losses.append(loss.item())
                optim.step()

                if stopping:
                    current_val_loss = self.validation(model, val_batches)
                    val_losses.append(current_val_loss)
                else:
                    if self.criterion in eb_criteria:
                        stop_trainig, val_losses = self.early_stopping(
                                                                        model=model,
                                                                        val_batches=val_batches,
                                                                        val_losses=val_losses,
                                                                        )
                        utils.clear_backprops(model)
                    elif self.criterion == 'training':
                        stop_trainig, val_losses = self.early_stopping(
                                                                        model=model,
                                                                        val_batches=val_batches,
                                                                        val_losses=val_losses,
                                                                        train_losses=train_losses,
                                                                        steps=steps,
                                                                        )
                    elif self.criterion == 'validation':
                        stop_trainig, val_losses = self.early_stopping(
                                                                        model=model,
                                                                        val_batches=val_batches,
                                                                        val_losses=val_losses,
                                                                        steps=steps,
                                                                        )
                    if stop_trainig:
                        stopping = steps
                steps += 1
                # self._save_params(model, optim, steps)
                #break
            if verbose:
                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                print(f'==== Steps: {steps}, Train loss: {avg_train_loss}, Val loss: {avg_val_loss} ====')
        return stopping, train_losses, val_losses
