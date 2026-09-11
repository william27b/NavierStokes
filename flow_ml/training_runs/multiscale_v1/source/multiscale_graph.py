"""Static, geometry-only graphs; connected aggregation preserves solid barriers."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

SCALE = np.array([.05, .05, .0025])
OFFSET = np.array([0., 0., 1/3])
FEATURES = ('x', 'y', 'inlet', 'outlet', 'solid_left', 'solid_right',
            'solid_below', 'solid_above', 'wall_distance', 'inlet_ux', 'inlet_uy', 'viscosity')

def fluid_edges(solid):
    ids = np.full(solid.shape, -1, dtype=np.int64)
    ids[~solid] = np.arange((~solid).sum())
    edges = []
    for a, b in [(ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :])]:
        valid = (a >= 0) & (b >= 0)
        edges.append(np.stack((a[valid], b[valid])))
    forward = np.concatenate(edges, axis=1)
    return np.concatenate((forward, forward[::-1]), axis=1)

def connected_pool(positions, edges, block_size):
    """Only join nodes in the same spatial block along existing fluid edges."""
    cells = np.floor(positions / block_size).astype(np.int64)
    parent = np.arange(len(positions))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    a, b = edges
    inside = np.all(cells[a] == cells[b], axis=1)
    for i, j in edges[:, inside].T:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(parent))])
    _, mapping = np.unique(roots, return_inverse=True)
    coarse = mapping[edges]
    coarse = coarse[:, coarse[0] != coarse[1]]
    coarse = np.unique(coarse.T, axis=0).T
    return mapping, coarse

def numpy_pool(values, mapping, weights):
    sums = np.zeros((int(mapping.max())+1, values.shape[1]), dtype=np.float64)
    count = np.bincount(mapping, weights=weights)
    np.add.at(sums, mapping, values * weights[:, None])
    return sums / count[:, None], count

def build_graph(solid, inlet_velocity=(.01, 0.), tau=.53, levels=5):
    solid = np.asarray(solid, dtype=bool)
    ny, nx = solid.shape
    if not (~solid).any() or min(nx, ny) < 3 or levels < 1:
        raise ValueError('Graph needs fluid, dimensions >=3, and at least one level')
    yy, xx = np.nonzero(~solid)
    positions = np.stack((xx, yy), axis=1).astype(float)
    normalized = positions / [nx-1, ny-1]
    boundaries = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        xn, yn = xx+dx, yy+dy
        inside = (xn >= 0) & (xn < nx) & (yn >= 0) & (yn < ny)
        blocked = np.zeros(len(xx))
        blocked[inside] = solid[yn[inside], xn[inside]]
        boundaries.append(blocked)
    inlet = xx == 0
    outlet = xx == nx-1
    distance = distance_transform_edt(~solid)[~solid] if solid.any() else np.full(len(xx), max(nx, ny))
    x = np.column_stack((normalized, inlet, outlet, *boundaries, distance/(ny-2),
                         np.full(len(xx), inlet_velocity[0]/.05),
                         np.full(len(xx), inlet_velocity[1]/.05),
                         np.full(len(xx), ((tau-.5)/3)/.01)))
    bc_mask = np.zeros((len(xx), 3), dtype=bool)
    bc_mask[inlet, :2] = True
    bc_mask[outlet, 1:] = True  # This solver prescribes outlet uy=0 and rho=1.
    bc_values = np.zeros((len(xx), 3), dtype=np.float32)
    bc_values[inlet, :2] = np.asarray(inlet_velocity)/.05
    hierarchy = []
    weights = np.ones(len(xx))
    edges = fluid_edges(solid)
    features = x.copy()
    for level in range(levels):
        width = 2**level
        a, b = edges
        delta = (positions[b]-positions[a])/width
        geom = np.column_stack((features, weights/width**2))
        entry = {'edges': torch.tensor(edges, dtype=torch.long),
                 'edge_attr': torch.tensor(delta, dtype=torch.float32),
                 'geom': torch.tensor(geom, dtype=torch.float32),
                 'weights': torch.tensor(weights, dtype=torch.float32),
                 'positions': torch.tensor(positions, dtype=torch.float32)}
        if level < levels-1:
            mapping, coarse_edges = connected_pool(positions, edges, 2*width)
            entry['mapping'] = torch.tensor(mapping, dtype=torch.long)
            positions, next_weights = numpy_pool(positions, mapping, weights)
            features, _ = numpy_pool(features, mapping, weights)
            weights, edges = next_weights, coarse_edges
        hierarchy.append(entry)
    obstacles = solid.copy()
    obstacles[[0, -1], :] = False
    obstacle_distance = (distance_transform_cdt(~obstacles, metric='taxicab')[~solid]
                         if obstacles.any() else np.full(len(xx), nx+ny))
    return {'x': torch.tensor(x, dtype=torch.float32), 'levels': hierarchy,
            'bc_mask': torch.tensor(bc_mask), 'bc_values': torch.tensor(bc_values),
            'far_mask': torch.tensor(obstacle_distance > 8),
            'halo_mask': torch.tensor((obstacle_distance >= 6) & (obstacle_distance <= 10)),
            'solid': torch.tensor(solid)}

def load_graph(run_dir, levels=5):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir/'metadata.json').read_text())
    with np.load(run_dir/'fields.npz') as f:
        solid = f['solid']
        inlet = f['applied_u'][:, ~solid[:, 0], 0].mean(axis=1)
        graph = build_graph(solid, inlet, float(f['tau']), levels)
        raw = np.stack((f['u'][0], f['u'][1], f['pressure']), axis=-1)[~solid]
        graph['y'] = torch.tensor((raw-OFFSET)/SCALE, dtype=torch.float32)
    graph['seed'] = meta['generation']['seed']
    graph['run_dir'] = str(run_dir)
    return graph

def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k,v in value.items()}
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value

def pool(values, level, next_level):
    """Weighted mean over represented fluid cells, including irregular aggregates."""
    result = values.new_zeros((len(next_level['weights']), values.shape[-1]))
    result.index_add_(0, level['mapping'], values * level['weights'][:, None])
    return result / next_level['weights'][:, None]
