"""D2Q9 solid masks: each function returns a NumPy bool array shaped (ny, nx).

The top and bottom rows are always solid because the kernel uses channel walls.
Add a function with the same signature, then register it in SOLIDS.
"""
import numpy as np


def _grid(nx, ny):
    if nx < 8 or ny < 8:
        raise ValueError("Solid geometries need nx and ny of at least 8.")
    return np.indices((ny, nx))


def _walls(mask):
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def channel(nx, ny):
    """A straight channel with no interior obstacle."""
    _grid(nx, ny)
    return _walls(np.zeros((ny, nx), dtype=bool))


def cylinder(nx, ny):
    """The original circular obstacle, one third of the way downstream."""
    yy, xx = _grid(nx, ny)
    return _walls((xx - nx // 3)**2 + (yy - (ny - 1) / 2)**2 <= (ny / 10)**2)


def rectangle(nx, ny):
    """A centered rectangular obstacle."""
    yy, xx = _grid(nx, ny)
    return _walls((np.abs(xx - nx // 3) <= nx * 0.06)
                  & (np.abs(yy - (ny - 1) / 2) <= ny * 0.11))


def ellipse(nx, ny):
    """An ellipse with its long axis along the flow."""
    yy, xx = _grid(nx, ny)
    return _walls(((xx - nx // 3) / (nx * 0.15))**2
                  + ((yy - (ny - 1) / 2) / (ny * 0.07))**2 <= 1)


def diamond(nx, ny):
    """A diamond with pointed upstream and downstream faces."""
    yy, xx = _grid(nx, ny)
    return _walls(np.abs(xx - nx // 3) / (nx * 0.10)
                  + np.abs(yy - (ny - 1) / 2) / (ny * 0.14) <= 1)


def two_cylinders(nx, ny):
    """Two circular obstacles in tandem along the channel center."""
    yy, xx = _grid(nx, ny)
    radius = ny * 0.07
    mask = np.zeros((ny, nx), dtype=bool)
    for center_x in (nx * 0.30, nx * 0.64):
        mask |= (xx - center_x)**2 + (yy - (ny - 1) / 2)**2 <= radius**2
    return _walls(mask)


def staggered_cylinders(nx, ny):
    """Three circles alternating above and below the channel center."""
    yy, xx = _grid(nx, ny)
    mask = np.zeros((ny, nx), dtype=bool)
    for center_x, center_y in ((0.28, 0.36), (0.50, 0.64), (0.72, 0.36)):
        mask |= ((xx - center_x * (nx - 1))**2
                 + (yy - center_y * (ny - 1))**2 <= (ny * 0.065)**2)
    return _walls(mask)


def constriction(nx, ny):
    """Smooth wall bumps narrowing the middle of the channel."""
    yy, xx = _grid(nx, ny)
    height = ny * 0.12 * np.exp(-((xx - (nx - 1) / 2) / (nx * 0.14))**2)
    return _walls((yy <= height) | (yy >= ny - 1 - height))


SOLIDS = {
    "channel": channel,
    "cylinder": cylinder,
    "rectangle": rectangle,
    "ellipse": ellipse,
    "diamond": diamond,
    "two_cylinders": two_cylinders,
    "staggered_cylinders": staggered_cylinders,
    "constriction": constriction,
}
