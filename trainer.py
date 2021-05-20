import torch
import utils

from torch.optim import SGD, RMSprop, Adam, AdamW
from scipy.stats import linregress

optimizers = ['SGD', 'RMSprop', 'Adam', 'AdamW']
eb_criteria = ['gradients', 'hessian']

class Trainer(object):

    def __init__(
                 self,
                 lr:float,
                 batch_size:int,
                 max_iter:int,
                 steps:int,
                 criterion:str,
                 optimizer:str,
                 device:torch.device,
                 window_size:int=None,
        ):
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_iter
        self.steps = steps
        self.criterion = criterion
        self.device = device
        self.window_size = window_size

        assert optimizer in optimizers, f'Optimizer must be one of {optimizers}\n'

        self.optimizer = optimizer

    def get_optim(self, model):
        if self.optimizer == 'SGD':
            optim = SGD(model.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSprop':
            optim = RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        elif self.optimizer == 'Adam':
            optim = Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            optim = AdamW(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01))
        return optim

    def mse(self, y:torch.Tensor, y_hat:torch.Tensor) -> torch.Tensor:
        return (y - y_hat).mul(0.5).pow(2).mean()

    def validation(self, model, val_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]]):
        model.eval()
        val_losses = torch.zeros(len(val_batches))
        with torch.no_grad():
            for i, batch in enumerate(val_batches):
                batch = tuple(t.to(self.device) for t in batch)
                X, y = batch
                y_hat = model(X)
                loss = self.mse(y, y_hat)
                val_losses[i] += loss.item()
        avg_val_loss = torch.mean(val_losses).item()
        return avg_val_loss

    def early_stopping(
                       self,
                       model,
                       batch_size:int=None,
                       val_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]]=None,
                       val_losses:List[float]=None,
                       train_losses:List[float]=None,
                       epoch:int=None,
                       ):
        stop_trainig = False
        if self.criterion == 'gradients':
            assert batch_size, '\nWhen evaluating an evidence-based criterion, batch size is required\n'
            utils.compute_sample_grads(model)
            avg_grad, var_estimator = utils.get_means_and_vars(model)
            eb_criterion = utils.eb_criterion(avg_grad, var_estimator, batch_size)
            utils.clear_backprops(model)
            if eb_criterion > 0:
                stop_trainig = True
            return stop_trainig

        elif self.criterion == 'hessian':
            raise NotImplementedError

        elif self.criterion == 'validation':
            assert val_batches, '\nWhen evaluating model performance, a separate validation set is required\n'
            assert self.window_size, '\nWhen evaluating mse on the val set, window size parameter is required\n'
            avg_val_loss = self.validation(model, val_batches)
            val_losses.append(avg_val_loss)
            if (epoch + 1) > self.window_size:
                lmres = linregress(range(self.window_size), val_losses[(epoch + 1 - self.window_size):(epoch + 2)])
                if (lmres.slope > 0) or (lmres.pvalue > .1):
                    stop_trainig = True
            return stop_trainig, val_losses

        elif self.criterion == 'training':
            assert self.window_size, '\nWhen evaluating mse on the train set, window size parameter is required\n'
            if (epoch + 1) > self.window_size:
                lmres = linregress(range(self.window_size), train_losses[(epoch + 1 - self.window_size):(epoch + 2)])
                if (lmres.slope > 0) or (lmres.pvalue > .1):
                    stop_trainig = True
            return stop_trainig


    def fit(self, model, train_batches:Iterator[Tuple[torch.Tensor, torch.Tensor]], val_batches=None):
        model.to(self.device)
        optim = self.get_optim(model)
        #register forward and backward hooks, if stopping criterion is evidence-based
        if self.criterion in eb_criteria:
            model = utils.register_hooks(model)

        train_losses, val_losses = [], []
        iter = 0
        for epoch in range(self.max_epochs):
            model.train()
            batch_losses = torch.zeros(len(train_batches))
            for i, batch in enumerate(train_batches):
                optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                X, y = batch
                y_hat = model(X)
                loss = mse(y, y_hat)
                loss.backward()
                batch_losses[i] += loss.item()
                optim.step()
                iter += 1

            avg_train_loss = np.mean(batch_losses)
            train_losses.append(avg_train_loss)

            if self.criterion in eb_criteria:
                stop_trainig = self.early_stopping(model=model, batch_size=batch_size)
            elif self.criterion == 'training':
                stop_trainig = self.early_stopping(model=model, train_losses=train_losses, epoch=epoch)
            elif self.criterion == 'validation':
                stop_trainig, val_losses = self.early_stopping(model=model, val_batches=val_batches, val_losses=val_losses, epoch=epoch)

            if stop_trainig:
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optim_state_dict': optim.state_dict(),
                        }, os.path.join(model_dir, f'model_epoch{epoch+1:04d}.tar'))
                break

        return train_losses
