# from time import time
# import matplotlib.pyplot as plt
import torch
# import numpy as np
# sin, exp = np.sin, np.exp
from itertools import islice
from torch import Tensor
from torch.nn import Identity
from torch.utils import data
from torch.autograd import grad


##############################################################################
#               custom activation functions
##############################################################################

class AlgSigmoid(Identity):
    def __init__(self, r=4):
        Identity.__init__(self)
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        return x/(self.r+x**2)**.5


##############################################################################
#               dataset handling
##############################################################################

def arrays_to_dataset(x, f, batch_size=1, shuffle=False):
    xt, ft = Tensor(x), Tensor(f)
    dataset = data.TensorDataset(xt, ft)
    return data.DataLoader(dataset, batch_size, shuffle=shuffle)


def batch(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


##############################################################################
#               PINN utils
##############################################################################

def autodiffable_domain(domain, shape=None):
    return Tensor(domain).requires_grad_()


def autodiff(f, x, make_graph=True): return grad(f, x, torch.ones_like(f),
                                                 retain_graph=True,
                                                 create_graph=make_graph)[0]
