import torch
from torch.nn import Sequential
from torch import Tensor
import numpy as np
from time import time


##############################################################################
#               usual sampling & interpolation model
##############################################################################

class Interpol:
    def __init__(self, layers, label=None):
        # build neural network
        self.net = Sequential(*layers)
        self.label = label
        self.input_len = layers[0].in_features

    def __call__(self, x):
        # test if tensor
        if torch.is_tensor(x):
            return self.net(x).detach().numpy()
        # test if numpy array
        elif isinstance(x, np.ndarray):
            if x.shape == (self.input_len, ):
                return self.net(Tensor(x.astype('float64'))).detach().numpy()
            elif len(x[0]) == self.input_len and len(x) > 1:
                new_shape = (len(x), self.input_len)
                return self.net(Tensor(x.reshape(new_shape))).detach().numpy()

    def add_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, epochs, train_set, val_set, quiet=False):
        history = {}
        for step in range(epochs):
            t0 = time()
            train_loss, val_loss = self.compute_training_step(train_set,
                                                              val_set)
            history[f'Epoch {step+1}'] = {'tloss': train_loss,
                                          'vloss': val_loss}
            tf = time()
            if not quiet:
                print(f"Epoch {step+1}/{epochs}\n\n",
                      f"Training loss {train_loss}\n",
                      f"Validation loss {val_loss}\n",
                      "Time spent: {:.6f} seg\n".format(tf-t0),
                      "-------------------\n\n")
        return history

    def compute_training_step(self, train_set, val_set):
        # training
        train_loss, val_loss = 0, 0
        for x, f in train_set:
            # Compute prediction and loss
            pred = self.net(x)
            loss = self.loss_fn(pred, f)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        # validation
        with torch.no_grad():
            # Compute prediction and loss
            for x, f in val_set:
                pred = self.net(x)
                val_loss += self.loss_fn(pred, f).item()

        return train_loss, val_loss

    def save_nn(self, path):
        torch.save(self.net.state_dict(), path)
        pass

    def load_nn(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        pass


##############################################################################
#                PINN
##############################################################################

class PINN:
    def __init__(self, layers, label=None):
        # build neural network
        self.net = Sequential(*layers)
        self.input_len = layers[0].in_features

        # label
        self.label = label

    def __call__(self, x):
        # test if tensor
        if torch.is_tensor(x):
            return self.net(x).detach().numpy()
        # test if numpy array
        elif isinstance(x, np.ndarray):
            return self.net(
                Tensor(x.reshape(len(x), self.input_len))).detach().numpy()

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer

    def add_physics_loss(self, phys_loss):
        self.phys_loss = phys_loss

    def train(self, epochs, quiet=False):
        history = {}
        for step in range(1, epochs+1):
            t0 = time()
            train_loss = self.compute_training_step()
            history[f'Epoch {step}'] = {'tloss': train_loss}
            tf = time()
            if not quiet:
                print(f"Epoch {step}\n")
                print(f"Training loss {train_loss}")
                print("Time spent: {:.6f} seg".format(tf-t0))
                print("-------------------\n\n")
        return history

    def compute_training_step(self):
        # Compute physics informed loss
        loss = self.phys_loss(self)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()

        return train_loss

    def save_nn(self, path):
        torch.save(self.net.state_dict(), path)
        pass

    def load_nn(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
        pass
