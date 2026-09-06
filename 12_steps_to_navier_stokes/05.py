from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import pyplot, cm

nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)

u = np.ones((ny, nx))
u[int(.5 / dy): int(1 / dy + 1), int(.5 / dx): int(1 / dx + 1)] = 2

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
pyplot.savefig("05_surface_plot.png")

row, col = u.shape
# for n in range(nt + 1):
#     un = u.copy()
#     for j in range(1, row):
#         for i in range(1, col):
#             u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) - (c * dt / dy * (un[j, i] - un[j - 1, i])))

#             u[0, :] = 1
#             u[-1, :] = 1
#             u[:, 0] = 1
#             u[:, -1] = 1

# vectorized operations
for n in range(nt + 1):
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) - (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))

    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.add_subplot(projection='3d')
surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
pyplot.savefig("05_surface_plot_propagated.png")

