from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import pyplot, cm

nx = 31
ny = 31
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

x = np.linspace(0, 2, nx)
y = np.linspace(0, 1, ny)

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)

def plot(p, name):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    pyplot.savefig(name)

def laplace(nt):
    p = np.zeros((nx, ny))

    p[:, 0] = 0
    p[:, -1] = y
    p[0, :] = p[1, :]
    p[-1, :] = p[-2, :]

    plot(p, "09_initial.png")

    # vectorized operations
    for _ in range(nt):
        pn = p.copy()
        p[1:-1,1:-1] = ((dy**2) * (pn[1:-1, 2:] + pn[1:-1, :-2]) + (dx**2) * (pn[2:, 1:-1] + pn[:-2, 1:-1])) / (2 * (dx**2 + dy**2))

        p[:, 0] = 0
        p[:, -1] = y
        p[0, :] = p[1, :]
        p[-1, :] = p[-2, :]

        print(np.sum(np.abs(p) - np.abs(pn)) / np.sum(np.abs(pn)))

    plot(p, f"09_surface_plot_laplace_{nt}.png")

if __name__ == "__main__":
    laplace(500)
    # alternatively, the code can be altered to reach a l1 norm instead of a certain number of steps