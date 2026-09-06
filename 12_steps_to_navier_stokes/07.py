from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import pyplot, cm

nx = 31
ny = 31
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx): int(1 / dx + 1)] = 2

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
pyplot.savefig("07_surface_plot.png")

def diffuse(nt):
    row, col = u.shape

    # vectorized operations
    for n in range(nt + 1):
        un = u.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] + (nu * dt / dx**2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) + (nu * dt / dy**2) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
        
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    surf2 = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax.set_zlim(1, 2.5)
    pyplot.savefig(f"07_surface_plot_diffused_{nt}.png")

if __name__ == "__main__":
    diffuse(10)
    diffuse(14)
    diffuse(50)