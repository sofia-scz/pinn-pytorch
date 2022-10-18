from flexnnfw.core import Interpol
from flexnnfw.utils import AlgSigmoid
from torch.nn import Linear
import numpy as np
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# load model

h_nodes = 50
model = Interpol([Linear(2, h_nodes), AlgSigmoid(r=4),
                  Linear(h_nodes, h_nodes), AlgSigmoid(r=1),
                  Linear(h_nodes, 1)],
                 label='some weird function')

model.load_nn('ff_model')


# true function
def f(x, y):
    return (x+y)*np.exp(-.5*((x-1)**2+(y-1)**2))


def potential(q):
    x, y = q
    return (x+y)*np.exp(-.5*((x-1)**2+(y-1)**2))


nn_pot = model.__call__


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# simulations

# numerical solver
def rk4(f, t, x, dt):
    k1 = dt*f(t, x)
    k2 = dt*f(t+.5*dt, x+.5*k1)
    k3 = dt*f(t+.5*dt, x+.5*k2)
    k4 = dt*f(t+dt, x+k3)
    return x + (k1+k4+2*k2+2*k3)/6


def solve_ivp(f, x0, t0tf, dt):
    t, tf = t0tf
    x = x0
    result = [[t, *x]]
    while t < tf:
        x = rk4(f, t, x, dt)
        t = t + dt
        result.append([t, *x])
    return np.array(result)


def df(f, x, i, h=1e-3):
    a = np.zeros(len(x))
    a[i] += h
    return (f(x+a)-f(x-a))/2/h


def true_force(x):
    return np.array([df(potential, x, 0), df(potential, x, 1)])


def nn_force(x):
    return np.array([df(nn_pot, x, 0)[0], df(nn_pot, x, 1)[0]])


object_mass = .3


def tf_deriv(t, X):
    x, vx, y, vy = X
    pos = X[::2]
    fx, fy = true_force(pos)/object_mass
    return np.array([vx, fx, vy, fy])


def nnf_deriv(t, X):
    x, vx, y, vy = X
    pos = X[::2]
    fx, fy = nn_force(pos)/object_mass
    return np.array([vx, fx, vy, fy])


# iniconds
v = 1
x0, y0 = -5, -4
vx0, vy0 = v*1, v*.5
X0 = np.array([x0, vx0, y0, vy0])

t0tf = (0, 15)
dt = .1
solve_ref = solve_ivp(tf_deriv, X0, t0tf, dt)
true_x_tray = solve_ref[:, 1]
true_y_tray = solve_ref[:, 3]

solve_pred = solve_ivp(nnf_deriv, X0, t0tf, dt)
pred_x_tray = solve_pred[:, 1]
pred_y_tray = solve_pred[:, 3]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# plots
alen = 101
X = np.linspace(-7, 7, alen)
Y = X.copy()

# process results
R = np.array([(x, y) for x in X for y in Y])

F_ = nn_pot(R).reshape((alen, alen))

X, Y = np.meshgrid(X, Y)
F = f(X, Y)

true_Fmin, true_Fmax = F.min(), F.max()
nn_Fmin, nn_Fmax = F_.min(), F_.max()
vmin, vmax = min(nn_Fmin, true_Fmin), max(nn_Fmax, true_Fmax)

# plot some stuff
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

# plot results
map1 = axes[0].contourf(X, Y, F, levels=20, cmap='BuPu', vmin=vmin,
                        vmax=vmax)
axes[0].set_title('Validation')
axes[0].plot(true_x_tray, true_y_tray, color='#865', lw=1)
axes[0].scatter([true_x_tray[-1]], true_y_tray[-1], color='#865', s=10)
axes[0].plot(pred_x_tray, pred_y_tray, color='#c33', lw=1)
axes[0].scatter([pred_x_tray[-1]], pred_y_tray[-1], color='#c33', s=10)

map2 = axes[1].contourf(X, Y, F_, levels=20, cmap='BuPu', vmin=vmin,
                        vmax=vmax)
axes[1].set_title('Predicted')

for ax in axes:
    ax.set_aspect('equal', adjustable='box')

fig.colorbar(mappable=map1, ax=axes.ravel().tolist())
# end
plt.show()
