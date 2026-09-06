from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import pyplot, cm

nx = 41
ny = 41
nit = 50
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1
nu = 0.1
dt = .001

def plot(p, name):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    pyplot.savefig(name)

def poisson_boundaries(p):
    p[:, -1] = p[:, -2]
    p[:, 0] = p[:, 1]
    p[0, :] = p[1, :]
    p[-1, :] = 0

def poisson_step(p, u, v, rho, dt):
    ux = (u[1:-1,2:] - u[1:-1,:-2]) / (2*dx)
    uy = (u[2:,1:-1] - u[:-2,1:-1]) / (2*dy)
    vx = (v[1:-1,2:] - v[1:-1,:-2]) / (2*dx)
    vy = (v[2:,1:-1] - v[:-2,1:-1]) / (2*dy)

    b = rho * ((1 / dt) * (ux + vy) - (ux**2) - (2*uy*vx) - (vy**2))

    for _ in range(nit):
        pn = p.copy()

        p[1:-1,1:-1] = (
            (((pn[1:-1,2:] + pn[1:-1,:-2]) * (dy**2) + (pn[2:,1:-1] + pn[:-2,1:-1]) * (dx**2)) / (2 * (dx**2 + dy**2)))
            - (((dx**2) * (dy**2)) / (2 * (dx**2 + dy**2))) * b
        )

        poisson_boundaries(p)

def momentum_boundaries(u, v):
    u[0,:] = 0
    u[:,-1] = 0
    u[:,0] = 0
    u[-1,:] = 1

    v[-1,:] = 0
    v[0,:] = 0
    v[:,-1] = 0
    v[:,0] = 0

def momentum_step(p, u, v, rho, dt, nu):
    un = u.copy()
    vn = v.copy()

    u[1:-1,1:-1] = (
        un[1:-1,1:-1]
        - (un[1:-1,1:-1] * (dt / dx) * (un[1:-1,1:-1] - un[1:-1,:-2]))
        - (vn[1:-1,1:-1] * (dt / dy) * (un[1:-1,1:-1] - un[:-2,1:-1]))
        - ((dt / (rho * 2 * dx)) * (p[1:-1,2:] - p[1:-1,:-2]))
        + (nu * (
            (dt / (dx**2)) * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])
            + (dt / (dy**2)) * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]) 
        ))
    )

    v[1:-1,1:-1] = (
        vn[1:-1,1:-1]
        - (un[1:-1,1:-1] * (dt / dx) * (vn[1:-1,1:-1] - vn[1:-1,:-2]))
        - (vn[1:-1,1:-1] * (dt / dy) * (vn[1:-1,1:-1] - vn[:-2,1:-1]))
        - ((dt / (rho * 2 * dy)) * (p[2:, 1:-1] - p[:-2, 1:-1]))
        + (nu * (
            (dt / (dx**2)) * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])
            + (dt / (dy**2)) * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]) 
        ))
    )

    momentum_boundaries(u, v)

if __name__ == "__main__":
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    momentum_boundaries(u, v)
    poisson_boundaries(p)

    for _ in range(100):
        poisson_step(p, u, v, rho, dt)
        momentum_step(p, u, v, rho, dt, nu)

    plot(p, "11.png")

    fig = pyplot.figure(figsize=(11,7), dpi=100)
    # plotting the pressure field as a contour
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
    pyplot.colorbar()
    # plotting the pressure field outlines
    pyplot.contour(X, Y, p, cmap=cm.viridis)  
    # plotting velocity field
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2]) 
    pyplot.xlabel('X')
    pyplot.ylabel('Y');
    pyplot.savefig("11_countour_map.png")