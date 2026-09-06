import numpy as np
from matplotlib import pyplot
import time, sys

if __name__ == "__main__":
    nx = 41
    dx = 2 / (nx - 1)
    nt = 25
    dt = .025
    c = 1

    u = np.ones(nx)
    u[int(.5 / dx): int(1 / dx + 1)] = 2
    print(u)

    pyplot.plot(np.linspace(0, 2, nx), u)
    pyplot.savefig("02_plot_wave.png")

    for n in range(nt):
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] * (1 - dt / dx * (un[i] - un[i-1]))

    pyplot.close()
    pyplot.plot(np.linspace(0, 2, nx), u)
    pyplot.savefig("02_plot_wave_diffused.png")