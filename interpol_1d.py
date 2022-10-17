import numpy as np
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from flexnnfw.utils import arrays_to_dataset, AlgSigmoid
from flexnnfw.core import Interpol
from time import time
sin, exp = np.sin, np.exp

# torch
dtype = float
device = 'cpu'
model_in = 1
model_out = 1


# target function
def f(x): return exp(-.1*x**2)*sin(2*x)


# set up data
t_len, v_len = 50, 100
train_x, val_x = (np.linspace(-8, 8, t_len).reshape(t_len, model_in),
                  np.linspace(-10, 10, v_len).reshape(v_len, model_in))
train_y, val_y = f(train_x), f(val_x)

val_set = arrays_to_dataset(val_x, val_y, batch_size=v_len)

# set up training
training_configs = [{'lr': 1e-2, 'bs': 10, 'epochs': 1000},
                    {'lr': 1e-3, 'bs': 10, 'epochs': 300},
                    {'lr': 1e-4, 'bs': 10, 'epochs': 300}]


def train_model(model, training_configs):
    t0 = time()
    hists = []
    for config_dict in training_configs:
        lr = config_dict['lr']
        epochs = config_dict['epochs']
        batch_size = config_dict['bs']
        optimizer = optim.Adam(model.net.parameters(), lr=lr)
        model.add_optimizer(optimizer)
        train_set = arrays_to_dataset(train_x, train_y,
                                      batch_size=batch_size, shuffle=True)
        hists.append(model.train(epochs, train_set, val_set, quiet=False))

    tloss, vloss = [], []
    for hist in hists:
        for epoch in hist:
            d = hist[epoch]
            tloss.append(d['tloss'])
            vloss.append(d['vloss'])
    epochs = range(len(tloss))
    tf = time()
    print('Training finished. Time spent: {:.6f} seg.'.format(tf-t0))
    return epochs, tloss, vloss


h_nodes = 20
model = Interpol([nn.Linear(model_in, h_nodes), AlgSigmoid(r=4),
                  nn.Linear(h_nodes, h_nodes), AlgSigmoid(r=1),
                  nn.Linear(h_nodes, model_out)],
                 label='asig')

# set up loss
loss_fn = nn.L1Loss()
model.add_loss_fn(loss_fn)

# train model
epochs, tloss, vloss = train_model(model, training_configs)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
# plot training
axes[0].plot(epochs, tloss, label='training')
axes[0].plot(epochs, vloss, label='validation')
axes[0].set_title('Loss vs epoch')
axes[0].set_ylabel(model.label)
axes[0].set_yscale('log')

# plot results
axes[1].plot(val_x, model(val_x), color='#c33', label='predicted')
axes[1].scatter(val_x, val_y, color='#222', label='validation', s=10)

axes[1].xaxis.set_minor_locator(AutoMinorLocator())
axes[1].yaxis.set_minor_locator(AutoMinorLocator())

# group config
for ax in axes:
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', width=.8, length=4)
    ax.tick_params(which='minor', width=.6, length=3)
    ax.grid(color='grey', linestyle='-', linewidth=.5)
    ax.legend(fontsize=9)

# end
fig.tight_layout()
plt.show()
