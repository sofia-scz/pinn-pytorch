import numpy as np
from torch import nn, Tensor
import torch.optim as optim
import matplotlib.pyplot as plt
from flexnnfw.utils import arrays_to_dataset, AlgSigmoid
from flexnnfw.core import Interpol
from time import time

# torch
dtype = float
device = 'cpu'
model_in = 2
model_out = 1


# target function
def f(x, y): return (x+y)*np.exp(-.5*((x-1)**2+(y-1)**2))


# set up data
alen = 101
X = np.linspace(-5, 5, alen)
Y = X.copy()

# set up training data
training_samples = 5000
train_x, train_f = [], []
for _ in range(training_samples):
    i, j = np.random.randint(0, alen, 2)
    x, y = X[i], Y[j]
    train_x.append((x, y))
    train_f.append(f(x, y))
train_x = np.array(train_x)
train_f = np.array(train_f).reshape(training_samples, model_out)

val_x, val_f = [], []
for i in range(alen):
    for j in range(alen):
        x, y = X[i], Y[j]
        val_x.append((x, y))
        val_f.append(f(x, y))
val_x = np.array(val_x)
val_f = np.array(val_f).reshape(alen**2, model_out)

val_set = arrays_to_dataset(val_x, val_f, batch_size=alen**2)


# set up training
training_configs = [{'lr': 1e-2, 'bs': 1000, 'epochs': 600},
                    {'lr': 1e-3, 'bs': 1000, 'epochs': 2000},
                    {'lr': 1e-4, 'bs': 1000, 'epochs': 200},
                    {'lr': 1e-5, 'bs': 1000, 'epochs': 200}]


def train_model(model, training_configs):
    t0 = time()
    hists = []
    for config_dict in training_configs:
        lr = config_dict['lr']
        epochs = config_dict['epochs']
        batch_size = config_dict['bs']
        optimizer = optim.Adam(model.net.parameters(), lr=lr)
        model.add_optimizer(optimizer)
        train_set = arrays_to_dataset(train_x, train_f,
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
    print('Training finished. Time spent: {:.2f} seg.'.format(tf-t0))
    return epochs, tloss, vloss


h_nodes = 50
model = Interpol([nn.Linear(model_in, h_nodes), AlgSigmoid(r=4),
                  nn.Linear(h_nodes, h_nodes), AlgSigmoid(r=1),
                  nn.Linear(h_nodes, model_out)],
                 label='some weird function')


# set up loss
loss_fn = nn.L1Loss()
model.add_loss_fn(loss_fn)

# train model
epochs, tloss, vloss = train_model(model, training_configs)
model.save_nn('ff_model')

# process results

R = np.array([(x, y) for x in X for y in Y])

F_ = model(R).reshape((alen, alen))

X, Y = np.meshgrid(X, Y)
F = f(X, Y)

true_Fmin, true_Fmax = F.min(), F.max()
nn_Fmin, nn_Fmax = F_.min(), F_.max()
vmin, vmax = min(nn_Fmin, true_Fmin), max(nn_Fmax, true_Fmax)

# plot some stuff
fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=150)
# plot training
axes[0].plot(epochs, tloss, label='training')
axes[0].plot(epochs, vloss, label='validation')
axes[0].set_title('Loss vs epoch')
axes[0].set_ylabel(model.label)
axes[0].set_yscale('log')
axes[0].legend()

# plot results
map1 = axes[1].contourf(X, Y, F, levels=20, cmap='viridis', vmin=vmin,
                        vmax=vmax)
axes[1].set_title('Validation')

map2 = axes[2].contourf(X, Y, F_, levels=20, cmap='viridis', vmin=vmin,
                        vmax=vmax)
axes[2].set_title('Predicted')

for ax in axes[1:]:
    ax.set_aspect('equal', adjustable='box')

fig.colorbar(mappable=map1, ax=axes.ravel().tolist())
# end
plt.show()
