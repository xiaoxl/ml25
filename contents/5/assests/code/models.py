import torch
import numpy as np
import random
import time


class ModelTemplate():
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.stats = {'losses': [],
                      'val_losses': [],
                      'train_time': [],
                      'val_time': [],
                      'n_epochs': 0}

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def save(self, filename='model.pth'):
        model_state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats
        }
        torch.save(model_state, filename)

    def load(self, filename='model.pth'):
        model_state = torch.load(filename, weights_only=False)
        self.model.load_state_dict(model_state['model'])
        self.optimizer.load_state_dict(model_state['optimizer'])
        self.stats = model_state['stats']

        self.model.train()

    def log_update(self, train_time, loss, val_time, val_loss):
        self.stats['train_time'].append(train_time)
        self.stats['losses'].append(loss)
        self.stats['val_time'].append(val_time)
        self.stats['val_losses'].append(val_loss)
        self.stats['n_epochs'] += 1

    def log_output(self, verbose=1, formatstr=''):
        s = [f'epoch {self.stats['n_epochs']}',
             f'train_time: {{{formatstr}}}'.format(self.stats['train_time'][-1]),
             f'loss: {{{formatstr}}}'.format(self.stats['losses'][-1])]
        if self.stats['val_losses'][-1] is not None:
            s.append(f'val_time: {{{formatstr}}}'.format(self.stats['val_time'][-1]))
            s.append(f'val_loss: {{{formatstr}}}'.format(self.stats['val_losses'][-1]))
        if verbose == 1:
            print(' '.join(s))
        return s

    def _train_one_epoch(self, dataloader):
        self.model.train()

        losses = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            yhat = self.model(X_batch)
            loss = self.loss_fn(yhat, y_batch)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
        return np.mean(losses)

    def _eval_one_epoch(self, dataloader):
        self.model.eval()
        losses = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            yhat = self.model(X_batch)
            loss = self.loss_fn(yhat, y_batch)
            losses.append(loss.item())
        return np.mean(losses)

    def train(self, train_loader, val_loader=None, epoch_num=10, verbose=0, SEED=42):
        self.set_seed(SEED)
        for _ in range(epoch_num):
            start_time = time.time()
            loss = self._train_one_epoch(train_loader)
            end_time = time.time()
            train_time = end_time - start_time

            val_loss = None
            val_time = None
            if val_loader is not None:
                start_time = time.time()
                val_loss = self._eval_one_epoch(val_loader)
                end_time = time.time()
                val_time = end_time - start_time

            self.log_update(train_time, loss, val_time, val_loss)
            self.log_output(verbose=verbose)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.as_tensor(X, dtype=float)
        y_tensor = self.model(X_tensor.to(self.device))
        self.model.train()
        y = y_tensor.detach().cpu().numpy()

        return y
