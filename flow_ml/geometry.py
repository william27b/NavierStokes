"""NumPy-only obstacle masks.

Array shape is (ny, nx); center is (x, y), in lattice-cell coordinates.
Integer coordinates denote cell centers. Rotation is in radians, with positive
angles rotating +x toward +y. The result is clipped to the supplied grid.
These functions draw individual obstacles; channel walls belong to sample_case.
"""

import numpy as np


def convex_hull(points):
    """Return the convex hull of (N, 2) points, counterclockwise.

    Andrew's monotone-chain algorithm: O(N log N) including sorting.
    Duplicates and collinear points between hull endpoints are discarded.
    Degenerate inputs return zero, one, or two points, without a closing repeat.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")

    # unique also sorts the rows lexicographically by x, then y.
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points.copy()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(sequence):
        hull = []
        for point in sequence:
            # A clockwise turn or a straight continuation cannot be a hull corner.
            while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                hull.pop()
            hull.append(point)
        return hull

    lower = half(points)
    upper = half(points[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def _local_grid(shape, center, radius, axis_scale, rotation):
    """Express pixel centers in a rotated, scaled coordinate system."""
    shape = np.asarray(shape)
    if shape.shape != (2,) or shape.dtype.kind not in "iu" or np.any(shape <= 0):
        raise ValueError("shape must contain positive integers (ny, nx)")

    center = np.asarray(center, dtype=np.float64)
    axis_scale = np.asarray(axis_scale, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center must contain finite coordinates (x, y)")
    if axis_scale.shape != (2,) or not np.isfinite(axis_scale).all() or np.any(axis_scale <= 0):
        raise ValueError("axis_scale must contain two positive finite values")
    if not np.isfinite(radius) or radius <= 0 or not np.isfinite(rotation):
        raise ValueError("radius must be positive and finite; rotation must be finite")

    yy, xx = np.ogrid[:int(shape[0]), :int(shape[1])]
    dx, dy = xx - center[0], yy - center[1]
    cosine, sine = np.cos(rotation), np.sin(rotation)
    # Inverse rotation followed by inverse scaling.
    x = (cosine * dx + sine * dy) / (radius * axis_scale[0])
    y = (-sine * dx + cosine * dy) / (radius * axis_scale[1])
    return x, y


def convex_mask(
    shape, center, radius, *, rng, n_points=4, irregularity=0.35,
    axis_scale=(1.0, 1.0), rotation=0.0, normalize_area=False,
):
    """Draw a random convex polygon and return a boolean (ny, nx) mask.

    By default, radius bounds sampled vertices before axis scaling.
    normalize_area=True instead makes radius the radius of an equal-area disk:
    continuous polygon area = pi * radius**2 * axis_scale[0] * axis_scale[1],
    before clipping to the grid. Pixel counts approximate that continuous area.
    irregularity in [0, 1) varies radii and angles; zero gives a regular polygon.
    n_points is the number of candidates, not necessarily the final hull size.
    rng must supply NumPy Generator's uniform method.

    The continuous polygon contains center, though a subpixel or off-grid
    polygon can rasterize to an empty mask.
    """
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(n_points, (int, np.integer)) or n_points < 3:
        raise ValueError("n_points must be an integer >= 3")
    if not np.isfinite(irregularity) or not 0 <= irregularity < 1:
        raise ValueError("irregularity must be in [0, 1)")
    if not isinstance(normalize_area, (bool, np.bool_)):
        raise ValueError("normalize_area must be a boolean")
    x, y = _local_grid(shape, center, radius, axis_scale, rotation)

    spacing = 2 * np.pi / n_points
    phase = rng.uniform(0, 2 * np.pi)
    # For triangles, keep every angular gap below pi so center stays inside.
    jitter_limit = 0.24 if n_points == 3 else 0.35
    jitter = rng.uniform(-jitter_limit, jitter_limit, size=n_points) * irregularity
    angles = phase + spacing * (np.arange(n_points) + jitter)
    radii = rng.uniform(1 - irregularity, 1, size=n_points)
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    hull = convex_hull(points)
    if normalize_area:
        # Shoelace area in local coordinates; scaling lengths by s scales area by s**2.
        hx, hy = hull.T
        area = 0.5 * abs(np.dot(hx, np.roll(hy, -1)) - np.dot(hy, np.roll(hx, -1)))
        if not np.isfinite(area) or area <= 0:
            raise ValueError("Cannot normalize a degenerate convex hull")
        hull *= np.sqrt(np.pi / area)

    # For a CCW convex polygon, an interior point lies left of every edge.
    mask = np.ones(x.shape, dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
        mask &= cross >= -1e-12
    return mask


def organic_blob_mask(
    shape, center, radius, *, rng, roughness=0.30, harmonics=6,
    smoothness=2.0, axis_scale=(1.0, 1.0), rotation=0.0,
):
    """Draw a smooth, star-shaped radial blob as a boolean (ny, nx) mask.

    In local coordinates, the boundary is r(theta) = 1 + roughness * g(theta),
    with g a periodic Fourier sum normalized to stay in [-1, 1].
    Thus roughness in [0, 1) keeps every boundary radius positive.
    Zero roughness produces a disk (an ellipse when axis_scale differs).

    More harmonics allow finer features. Larger smoothness suppresses high
    frequencies. center is the polar origin, not necessarily the area centroid.
    Before axis scaling, radii stay within radius * (1 +/- roughness).
    Continuous star-shapedness does not guarantee pixel connectivity at every
    resolution; sample_case can apply its connectivity policy after combining masks.
    """
    if not np.isfinite(roughness) or not 0 <= roughness < 1:
        raise ValueError("roughness must be in [0, 1)")
    if isinstance(harmonics, (bool, np.bool_)) or not isinstance(harmonics, (int, np.integer)) or harmonics < 1:
        raise ValueError("harmonics must be a positive integer")
    if not np.isfinite(smoothness) or smoothness < 0:
        raise ValueError("smoothness must be finite and nonnegative")
    x, y = _local_grid(shape, center, radius, axis_scale, rotation)
    distance = np.hypot(x, y)
    if roughness == 0:
        return distance <= 1 + 1e-12

    theta = np.arctan2(y, x)
    frequencies = np.arange(1, harmonics + 1, dtype=np.float64)
    weights = rng.uniform(0.5, 1.0, size=harmonics) / frequencies**smoothness
    weights /= weights.sum()
    phases = rng.uniform(0, 2 * np.pi, size=harmonics)

    displacement = np.zeros_like(theta)
    for frequency, weight, phase in zip(frequencies, weights, phases):
        displacement += weight * np.cos(frequency * theta + phase)

    boundary_radius = 1 + roughness * displacement
    return distance <= boundary_radius + 1e-12
