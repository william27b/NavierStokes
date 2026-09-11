from pathlib import Path
import os

import numpy as np
import torch

def load_case(run_dir):
    with np.load(run_dir / "fields.npz") as fields:
        f           = fields['f']
        rho         = fields['rho']
        u           = fields['u']
        pressure    = fields['pressure']
        solid       = fields['solid']
        applied_u   = fields['applied_u']
        solid_name  = fields['solid_name']
        nx          = fields['nx']
        ny          = fields['ny']
        tau         = fields['tau']
        steps       = fields['steps']
        max_steps   = fields['max_steps']
        converged   = fields['converged']
        stop_reason = fields['stop_reason']

    fluid = ~solid

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    positions = np.stack((X, Y), axis=-1, dtype=np.float32)[fluid]
    targets = np.stack((u[0] / .05, u[1] / .05, (pressure - 1/3) / .0025), axis=2, dtype=np.float32)
    targets = targets[fluid]

    return positions, targets, solid, nx, ny

def build_edges(solid, nx, ny):
    fluid = ~solid
    node_id = np.full(solid.shape, -1, dtype=np.int64)
    node_id[fluid] = np.arange(fluid.sum(), dtype=np.int64)

    edges = []

    for y in range(ny):
        for x in range(nx):
            if not fluid[y][x]: continue

            if y > 0    and fluid[y-1][x  ]:
                edges.append((node_id[y,x], node_id[y-1, x  ]))

            if y < ny-1 and fluid[y+1][x  ]:
                edges.append((node_id[y,x], node_id[y+1, x  ]))

            if x > 0    and fluid[y  ][x-1]:
                edges.append((node_id[y,x], node_id[y  , x-1]))

            if x < nx-1 and fluid[y  ][x+1]:
                edges.append((node_id[y,x], node_id[y  , x+1]))

    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T

    return edge_index

def _validate_build_edges():
    test_1 = np.asarray([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.bool_)

    edge_index = build_edges(test_1, 3, 3)
    assert edge_index.shape == (2, 24)
    assert edge_index.dtype == np.int64

    test_2 = np.asarray([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.bool_)

    edge_index = build_edges(test_2, 3, 3)
    assert edge_index.shape == (2, 16)
    assert edge_index.dtype == np.int64

def to_torch(positions, targets, edge_index, nx, ny, device):
    normalized_positions = torch.as_tensor(positions)
    normalized_positions = normalized_positions / torch.as_tensor([nx-1, ny-1])
    # print(normalized_positions)

    src, dst = edge_index
    edge_attr = positions[dst] - positions[src]

    inputs = {
        "x": torch.as_tensor(normalized_positions, dtype=torch.float32, device=device),
        "edge_index": torch.as_tensor(edge_index, dtype=torch.long, device=device),
        "edge_attr": torch.as_tensor(edge_attr, dtype=torch.float32, device=device),
        "y": torch.as_tensor(targets, dtype=torch.float32, device=device)
    }

    # print(f"x:          {inputs['x'].shape}")
    # print(f"edge_index: {inputs['edge_index'].shape}")
    # print(f"edge_attr:  {inputs['edge_attr'].shape}")
    # print(f"y:          {inputs['y'].shape}")

    return inputs

if __name__ == "__main__":
    _validate_build_edges()