import numpy as np
from matplotlib import pyplot
import time, sys

if __name__ == "__main__":
    nx = 41
    dx = 2 / (nx - 1)
    nt = 20
    nu = 0.3 # viscosity
    sigma = 0.2
    dt = sigma * dx ** 2 / nu

    u = np.ones(nx)
    u[int(.5 / dx): int(1 / dx + 1)] = 2
    print(u)

    pyplot.plot(np.linspace(0, 2, nx), u)
    pyplot.savefig("03_plot_wave.png")

    for n in range(nt):
        un = u.copy()
        for i in range(1, nx-1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])

    pyplot.close()
    pyplot.plot(np.linspace(0, 2, nx), u)
    pyplot.savefig("03_plot_wave_propagated.png")