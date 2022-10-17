import torch
from torch.nn import Linear, Tanh, MSELoss
from torch.optim import RAdam
from flexnnfw.core import PINN
from flexnnfw.utils import autodiffable_domain, autodiff  # , AlgSigmoid
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from time import time
exp, cos = np.exp, np.cos


# torch
dtype = float

# define model
model_in = 1
model_out = 1

hidden_nodes = 100
model = PINN([Linear(model_in, hidden_nodes),
              Linear(hidden_nodes, hidden_nodes), Tanh(),
              Linear(hidden_nodes, hidden_nodes), Tanh(),
              Linear(hidden_nodes, model_out)],
             label='50x3 model')

# set up training
training_configs = [{'lr': 1e-3, 'epochs': 10000},
                    {'lr': 1e-4, 'epochs': 2000000}, ]


def train_model(model, training_configs):
    t0 = time()
    hists = []
    for config_dict in training_configs:
        lr = config_dict['lr']
        epochs = config_dict['epochs']
        optimizer = RAdam(model.net.parameters(), lr=lr)
        model.add_optimizer(optimizer)
        hists.append(model.train(epochs, quiet=False))

    tloss = []
    for hist in hists:
        for epoch in hist:
            d = hist[epoch]
            tloss.append(d['tloss'])
    epochs = range(len(tloss))
    tf = time()
    print('Training finished. Time spent: {:.6f} seg.'.format(tf-t0))
    return epochs, tloss


# loss computing
loss_fn = MSELoss()   # nn.MSELoss()


# training domain arrays
x0, xf, npoints = 0, 25, 200
x_array = np.linspace(x0, xf, npoints)

# store domains in model
model.x_tensor = autodiffable_domain(x_array).reshape(-1, 1)


# define physics loss
def phys_loss(self):
    x = self.x_tensor
    f = self.net(x)

    dx = autodiff(f, x)
    ddx = autodiff(dx, x, make_graph=False)

    zero_tensor = x*0
    ode_diff = loss_fn(ddx+.3*dx+f, zero_tensor)

    zero_tensor = x[0]*0
    bconds_diff = loss_fn(f[0]-1, zero_tensor) + loss_fn(dx[0], zero_tensor)
    return ode_diff/npoints + bconds_diff/2


model.add_physics_loss(phys_loss)

# train the model
epochs, tloss = train_model(model, training_configs)

# analyze results
x_dom = np.linspace(x0, xf+5, 100)
true_f = exp(-.15*x_dom)*cos(.988686*x_dom)

# plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
# plot training
axes[0].plot(epochs, tloss, label='training')
axes[0].set_title('Loss vs epoch')
axes[0].set_ylabel(model.label)
axes[0].set_yscale('log')

# plot results
axes[1].scatter(x_dom, true_f, color='#444', label='true', s=5)
axes[1].plot(x_dom, model(x_dom), color='#c33', label='predicted', lw=2)
axes[1].set_title('diff eqn solution')

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
