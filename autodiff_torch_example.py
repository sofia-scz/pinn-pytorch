import torch
from torch.nn import Sequential, Linear, Tanh
from torch.autograd import grad


def autodiff(f, x, make_graph=True): return grad(f, x, torch.ones_like(f),
                                                 retain_graph=True,
                                                 create_graph=make_graph)[0]


# set up neural network
net = Sequential(Linear(2, 5), Tanh(), Linear(5, 1))

# set up working domain
x = torch.linspace(0, 1, 5)
y = torch.linspace(0, 4, 5)

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
f = net(xy)

# differentiate results
dx = autodiff(f, x_grid)
dy = autodiff(f, y_grid)

ddx = autodiff(dx, x_grid, make_graph=False)
ddy = autodiff(dy, y_grid, make_graph=False)
