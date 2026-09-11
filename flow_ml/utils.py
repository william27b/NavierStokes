import matplotlib.pyplot as plt
import numpy as np

def plot(values, solid, nx, ny, title=None, save=None):
    ux, uy, p = values[:,:,0], values[:,:,1], values[:,:,2]

    ux = np.ma.array(ux, mask=solid)
    uy = np.ma.array(uy, mask=solid)
    p = np.ma.array(p, mask=solid)
    y, x = np.indices((ny, nx))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    field = ax.imshow(p, origin="lower", cmap=cmap, interpolation="nearest")
    fig.colorbar(field, ax=ax, label="Pressure (lattice units)")
    ax.set(xlabel="x (lattice cells)", ylabel="y (lattice cells)", aspect="equal")

    if not title is None:
        ax.set_title(title, loc="left", pad=14)

    ax.streamplot(x, y, ux, uy, color="#daf5e8", density=1.65, linewidth=0.45, arrowsize=0.55, maxlength=5, zorder=2)

    if not save is None:
        plt.savefig(save)

    return fig, ax