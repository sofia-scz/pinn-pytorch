import torch
from torch import nn, Tensor
from torch.optim import RAdam
from flexnnfw.core import PINN
from flexnnfw.utils import autodiffable_domain, autodiff
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
from time import time
exp, cos = np.exp, np.cos


# torch
dtype = float
device = 'cpu'

# define model
model_in = 2
model_out = 1

hidden_nodes = 100
model = PINN([nn.Linear(model_in, hidden_nodes),
              nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(),
              nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(),
              nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(),
              nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(),
              nn.Linear(hidden_nodes, hidden_nodes), nn.Tanh(),
              nn.Linear(hidden_nodes, model_out)],
             label='100x5 model')

# set up training
training_configs = [{'lr': 1e-4, 'epochs': 10000}, ]


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
loss_fn = nn.MSELoss()   # nn.MSELoss()


# domain arrays
x0, xf, nx = -5, 5, 40
y0, yf, ny = -5, 5, 40
x_array = np.linspace(x0, xf, nx)
y_array = np.linspace(y0, yf, ny)


# target function
def target_f(x, y): return (2*x*y)/(x**4+y**4+10)


def lap_f(x, y): return (8*x*y*(3*x**6 - 5*x**4*y**2 - 5*x**2*(10 + y**4)
                                + y**2*(-50 + 3*y**4)))/(10 + x**4 + y**4)**3


# store domains in model
model.x_domain = autodiffable_domain(x_array)
model.y_domain = autodiffable_domain(y_array)


def ss(a):
    x, y = a.detach().numpy()
    return lap_f(x, y)


def source(A):
    return Tensor([ss(a) for a in A]).reshape(-1, 1)


x = np.linspace(x0, xf, nx)
y = np.linspace(y0, yf, ny)
X, Y = np.meshgrid(x, y)
target = target_f(X, Y)
tbound = np.concatenate((target[:, 0], target[-1, :], target[:, -1],
                         target[0, :]))


# define physics loss
def phys_loss(self):
    x = self.x_domain
    y = self.y_domain

    # make grid
    x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
    x_grid.requires_grad_()
    y_grid.requires_grad_()
    x_grid.retain_grad()
    y_grid.retain_grad()

    # flaten grid arrays
    x_grid, y_grid = x_grid.flatten(), y_grid.flatten()

    # make one array
    xy = torch.stack((x_grid, y_grid)).transpose(0, 1)

    # compute model over the grid
    f = self.net(xy)

    # differentiate results
    dx = autodiff(f, x_grid)
    dy = autodiff(f, y_grid)

    ddx = autodiff(dx, x_grid, make_graph=False)
    ddy = autodiff(dy, y_grid, make_graph=False)

    # poisson pde diff
    pde_diff = loss_fn(ddx+ddy-Tensor(source(xy)).reshape(-1, 1), 0*f)

    # boundary conditions diff
    fmatrix = f.reshape(ny, nx)
    fbound = torch.cat((fmatrix[:, 0], fmatrix[-1, :], fmatrix[:, -1],
                        fmatrix[0, :]))
    bconds_diff = loss_fn(fbound - Tensor(tbound), 0*fbound)
    return pde_diff/nx/ny + bconds_diff/2/(nx+ny)


model.add_physics_loss(phys_loss)

# train the model
epochs, tloss = train_model(model, training_configs)

# ~~~~~~~~~~~~~~~~~           plots         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
npoints = 100
# check stuff
x = np.linspace(x0, xf, npoints)
y = np.linspace(y0, yf, npoints)
X, Y = np.meshgrid(x, y)
F = target_f(X, Y)

# plot target
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)
#
mappable = axes[0].contourf(X, Y, F, levels=30, cmap='coolwarm')
axes[0].set_title('Target function')
fig.colorbar(mappable=mappable, ax=axes[0])
#
mappable = axes[1].contourf(X, Y, lap_f(X, Y), levels=30, cmap='PuOr_r')
axes[1].set_title('Target laplacian')
fig.colorbar(mappable=mappable, ax=axes[1])
#
for ax in axes:
    ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()


# ~~~~~~~~~~~~~~~~~           results         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
XY = np.array([(xx, Y.flatten()[i]) for i, xx in enumerate(X.flatten())])
F_ = model(XY).reshape(npoints, npoints)

true_Fmin, true_Fmax = F.min(), F.max()
nn_Fmin, nn_Fmax = F_.min(), F_.max()
vmin, vmax = min(nn_Fmin, true_Fmin), max(nn_Fmax, true_Fmax)

# plots
fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=150,
                         gridspec_kw={'width_ratios': [1.5, 1, 1]})
# plot training
axes[0].plot(epochs, tloss, label='training')
axes[0].set_title('Loss vs epoch')
axes[0].set_ylabel(model.label)
axes[0].set_yscale('log')
axes[0].grid(color='grey', linestyle='-', linewidth=.5)

# plot results
mappable = axes[1].contourf(X, Y, F_, levels=30, cmap='viridis', vmin=vmin,
                            vmax=vmax)
axes[1].set_title('diff eqn solution')
#
mappable = axes[2].contourf(X, Y, F, levels=30, cmap='viridis', vmin=vmin,
                            vmax=vmax)
axes[2].set_title('True')
#
# group config
for ax in axes[1:]:
    ax.set_aspect('equal', adjustable='box')

# end
fig.colorbar(mappable=mappable, ax=axes.ravel().tolist())
plt.show()
