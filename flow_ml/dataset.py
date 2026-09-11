from pathlib import Path
from datetime import datetime, timezone
import json
import os

import numpy as np
import torch

from scipy.ndimage import label
from geometry import convex_mask, organic_blob_mask

from simulation import D2Q9Solver

def _weighted_counts(values, weights, minimum, name):
    """Validate integer choices and normalize their relative weights."""
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu" or np.any(values < minimum):
        raise ValueError(f"{name} choices must be integers >= {minimum}")
    total = weights.sum()
    if (
        weights.shape != values.shape or not np.isfinite(weights).all()
        or np.any(weights < 0) or not np.isfinite(total) or total <= 0
    ):
        raise ValueError(f"{name} weights must match the choices and have a positive finite sum")
    return values, weights / total


def sample_case(rng, sampling_config):
    """Sample component masks and flow settings using only the supplied RNG.

    convex_probability is the probability of proposing a convex shape for each
    component; otherwise an organic blob is proposed. Components may overlap:
    component_count counts shape proposals, not distinct connected obstacles.
    Without a fixed count, component_count_choices/weights select it once per
    case. convex_point_choices/weights select candidate points per convex shape,
    unless convex_options explicitly supplies a fixed n_points.
    Their union is clipped to eligible cells, then channel walls are added.
    Reject the whole proposal if a component is empty after clipping or the
    resulting fluid is disconnected. max_attempts bounds this rejection loop.

    Return {"solid": mask, "flow": settings, "sampling": geometry metadata}.
    The caller owns the seed, simulation, and saving.
    """
    c = sampling_config
    for name, minimum in (
        ("nx", 3), ("ny", 3), ("boundary_clearance", 0),
        ("max_attempts", 1),
    ):
        value = c[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    nx, ny = int(c["nx"]), int(c["ny"])
    boundary_clearance = int(c["boundary_clearance"])
    if 2 * boundary_clearance >= nx:
        raise ValueError("boundary_clearance must leave room for obstacles")

    convex_probability = float(c["convex_probability"])
    if not np.isfinite(convex_probability) or not 0 <= convex_probability <= 1:
        raise ValueError("convex_probability must be in [0, 1]")

    position_mean = np.asarray([c["x_mean"], c["y_mean"]], dtype=float)
    position_std = np.asarray([c["x_std"], c["y_std"]], dtype=float)
    if not np.isfinite(position_mean).all() or not np.isfinite(position_std).all() or np.any(position_std < 0):
        raise ValueError("position means must be finite and standard deviations nonnegative")

    ranges = {}
    for name in ("radius_range", "ux_range", "tau_range"):
        values = np.asarray(c[name], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
            raise ValueError(f"{name} must contain finite, ordered (min, max) values")
        ranges[name] = values
    if ranges["radius_range"][0] <= 0:
        raise ValueError("radius_range must be positive")
    if ranges["tau_range"][0] <= 0.5:
        raise ValueError("tau_range must be greater than 0.5")
    uy = float(c.get("uy", 0.0))
    if not np.isfinite(uy):
        raise ValueError("uy must be finite")

    convex_options = {
        "irregularity": 0.35, "normalize_area": True,
        "axis_scale": (1.0, 1.0), "rotation": 0.0,
    }
    convex_options.update(c.get("convex_options", {}))
    organic_options = {
        "roughness": 0.30, "harmonics": 6, "smoothness": 2.0,
        "axis_scale": (1.0, 1.0), "rotation": 0.0,
    }
    organic_options.update(c.get("organic_options", {}))

    # A fixed component_count remains available for controlled tests.
    # Otherwise sample once per case, preserving that count across retries.
    if "component_count" in c:
        value = c["component_count"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("component_count must be an integer >= 0")
        component_count = int(value)
    else:
        count_choices, count_weights = _weighted_counts(
            c.get("component_count_choices", (1, 2, 3, 4)),
            c.get("component_count_weights", (45, 40, 12, 3)),
            0, "component_count",
        )
        component_count = int(rng.choice(count_choices, p=count_weights))

    if "n_points" in convex_options:
        value = convex_options["n_points"]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 3:
            raise ValueError("convex_options['n_points'] must be an integer >= 3")
    else:
        point_choices, point_weights = _weighted_counts(
            c.get("convex_point_choices", (3, 4, 5)),
            c.get("convex_point_weights", (45, 45, 10)),
            3, "convex points",
        )

    eligible = np.ones((ny, nx), dtype=bool)
    eligible[0, :] = False
    eligible[-1, :] = False
    eligible[:, :boundary_clearance] = False
    eligible[:, nx - boundary_clearance:] = False

    connectivity = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])

    for attempt in range(c["max_attempts"]):
        solid = np.zeros((ny, nx), dtype=bool)
        component_records = []

        for _ in range(component_count):
            position = rng.normal(position_mean, position_std)  # (x, y)
            radius = float(rng.uniform(*ranges["radius_range"]))
            if rng.random() < convex_probability:
                kind, generator = "convex", convex_mask
                options = dict(convex_options)
                if "n_points" not in options:
                    options["n_points"] = int(rng.choice(point_choices, p=point_weights))
            else:
                kind, generator, options = "organic", organic_blob_mask, organic_options

            component = generator(
                (ny, nx), center=position, radius=radius, rng=rng, **options,
            )
            component &= eligible
            if not component.any():
                break  # Retry the entire case rather than silently dropping a shape.
            solid |= component
            component_records.append({
                "kind": kind,
                "center": position.tolist(),
                "radius": radius,
                "options": dict(options),
            })

        if len(component_records) != component_count:
            continue

        solid[0, :] = True
        solid[-1, :] = True
        _, fluid_components = label(~solid, structure=connectivity)
        if fluid_components != 1:
            continue

        return {
            "solid": solid,
            "flow": {
                "ux": float(rng.uniform(*ranges["ux_range"])),
                "uy": uy,
                "tau": float(rng.uniform(*ranges["tau_range"])),
            },
            "sampling": {
                "method": "component_masks_v3",
                "component_count": component_count,
                "attempts": attempt + 1,
                "convex_probability": convex_probability,
                "boundary_clearance": boundary_clearance,
                "components": component_records,
            },
        }

    raise RuntimeError(
        f"Could not sample {component_count} nonempty components with connected "
        f"fluid after {c['max_attempts']} attempts; check positions, radii, and clearance."
    )

def generate_case(solver: D2Q9Solver, seed, config):
    try:
        result = sample_case(np.random.default_rng(seed), config)
    except RuntimeError:
        return None

    solid, flow, sampling = result['solid'], result['flow'], result['sampling']
    ux, uy, tau = flow['ux'], flow['uy'], flow['tau']

    descriptions = [
        f"convex ({component['options']['n_points']} points)"
        if component["kind"] == "convex" else "organic"
        for component in sampling["components"]
    ]
    description = ", ".join(descriptions)

    solve_result = solver.solve(solid, description, ux=ux, uy=uy, tau=tau, extra_metadata={
        "generation": {
            "seed": seed,
            "config": config,
            "sampling": sampling,
        }
    })
    return solve_result


def _next_generation_seed(root):
    """Continue after archived runs and attempted seeds in batch manifests."""
    next_seed = 1000
    for path in (root / "solver_runs").glob("*/metadata.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        seed = metadata.get("generation", {}).get("seed")
        if isinstance(seed, int):
            next_seed = max(next_seed, seed + 1)
    for path in root.glob("dataset_*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        next_seed = max(next_seed, manifest.get("next_seed", 1000))
        records = list(manifest.get("failed", []))
        for split_records in manifest.get("splits", {}).values():
            records.extend(split_records)
        for record in records:
            if isinstance(record.get("seed"), int):
                next_seed = max(next_seed, record["seed"] + 1)
    return next_seed


def _numpy_json_default(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def generate_cases(
    num_cases,
    *,
    start_seed=None,
    nx=64,
    ny=64,
    ux_range=(0.1, 0.1),
    tau_range=(0.8, 0.8),
    uy=0.0,
    convex_probability=0.5,
    radius_range=(4.0, 9.0),
    component_count_choices=(1, 2, 3, 4),
    component_count_weights=(45, 40, 12, 3),
    component_count=None,
    convex_point_choices=(3, 4, 5),
    convex_point_weights=(45, 45, 10),
    convex_options=None,
    organic_options=None,
    boundary_clearance=4,
    max_attempts=100,
    x_mean=None,
    y_mean=None,
    x_std=None,
    y_std=None,
    split_weights=(5, 1, 1),
    solver=None,
):
    """Generate and save a batch, returning its manifest and compact records.

    num_cases counts attempts, not guaranteed converged cases. Sampling and
    convergence failures are recorded without replacement. split_weights are
    relative train/validation/test weights; (5, 1, 1) gives 100/20/20 attempts
    for num_cases=140. Integer counts use largest remainders, ties in that order.

    Omit start_seed to continue after existing run and manifest seeds (starting
    at 1000). An explicit seed range allows replay; keep replays in the same
    split. This automatic seed selection is intended for sequential batches.

    Position defaults follow nx/ny; radii and clearance remain in lattice cells.
    Shape option dictionaries override the existing defaults. One solver is
    reused for the batch; an existing instance can be supplied. Archives go to
    solver_runs beside this module. A unique dataset_*.json manifest is updated
    after every attempt. Its splits contain only converged runs, and the return
    value includes the manifest_path. No field arrays accumulate in the return.
    """
    if isinstance(num_cases, (bool, np.bool_)) or not isinstance(num_cases, (int, np.integer)) or num_cases < 0:
        raise ValueError("num_cases must be a nonnegative integer")
    num_cases = int(num_cases)
    weights = np.asarray(split_weights, dtype=float)
    if (
        weights.shape != (3,) or not np.isfinite(weights).all()
        or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0
    ):
        raise ValueError("split_weights must contain three finite nonnegative weights with a positive sum")
    raw_counts = num_cases * (weights / weights.sum())
    counts = np.floor(raw_counts).astype(int)
    remainder_order = np.argsort(-(raw_counts - counts), kind="stable")
    counts[remainder_order[:num_cases - int(counts.sum())]] += 1
    split_names = ("train", "validation", "test")

    root = Path(__file__).resolve().parent
    if start_seed is None:
        start_seed = _next_generation_seed(root)
    if isinstance(start_seed, (bool, np.bool_)) or not isinstance(start_seed, (int, np.integer)) or start_seed < 0:
        raise ValueError("start_seed must be a nonnegative integer or None")
    start_seed = int(start_seed)
    config = {
        "nx": nx, "ny": ny,
        "ux_range": ux_range, "tau_range": tau_range, "uy": uy,
        "convex_probability": convex_probability,
        "radius_range": radius_range,
        "component_count_choices": component_count_choices,
        "component_count_weights": component_count_weights,
        "convex_point_choices": convex_point_choices,
        "convex_point_weights": convex_point_weights,
        "convex_options": {
            "irregularity": 0.35, "normalize_area": True,
            **(convex_options or {}),
        },
        "organic_options": {
            "roughness": 0.4, "harmonics": 6, "smoothness": 1.5,
            **(organic_options or {}),
        },
        "boundary_clearance": boundary_clearance,
        "max_attempts": max_attempts,
        "x_mean": nx / 2 if x_mean is None else x_mean,
        "y_mean": ny / 2 if y_mean is None else y_mean,
        "x_std": nx / 6 if x_std is None else x_std,
        "y_std": ny / 6 if y_std is None else y_std,
    }
    if component_count is not None:
        config["component_count"] = component_count
    # Normalize NumPy scalars/arrays before passing provenance to the solver.
    config = json.loads(json.dumps(config, default=_numpy_json_default))

    started_at = datetime.now(timezone.utc)
    manifest_path = root / f"dataset_{started_at:%Y%m%dT%H%M%S%fZ}.json"
    manifest = {
        "started_at": started_at.isoformat(),
        "manifest_path": str(manifest_path),
        "run_root": "solver_runs",
        "config": config,
        "num_cases": num_cases,
        "attempted": 0,
        "start_seed": start_seed,
        "next_seed": start_seed,
        "split_weights": weights.tolist(),
        "requested_counts": dict(zip(split_names, counts.tolist())),
        "splits": {split: [] for split in split_names},
        "failed": [],
    }
    with manifest_path.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)

    if solver is None and num_cases:
        solver = D2Q9Solver()
    seed = start_seed
    for split, count in zip(split_names, counts):
        for _ in range(int(count)):
            result = generate_case(solver, seed, config)
            if result is None:
                manifest["failed"].append({
                    "split": split, "seed": seed, "failed": "sampling",
                })
            else:
                record = {
                    "seed": seed,
                    "run_id": result["metadata"]["run_id"],
                    "steps": result["steps"],
                }
                if result["converged"]:
                    manifest["splits"][split].append({**record, "status": "converged"})
                else:
                    manifest["failed"].append({
                        **record, "split": split, "failed": "convergence",
                        "stop_reason": result["stop_reason"],
                    })
            del result
            seed += 1
            manifest["attempted"] += 1
            manifest["next_seed"] = seed
            # Replace a complete JSON file so an interrupted write leaves the
            # previous completed attempt available for inspection.
            temporary_path = manifest_path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_path.replace(manifest_path)

    for split, records in manifest["splits"].items():
        print(f"{split}: {len(records)} converged / {manifest['requested_counts'][split]} attempted")
    print(f"Failed: {len(manifest['failed'])}")
    print(f"Dataset manifest: {manifest_path}")
    return manifest


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

from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, run_dirs):
        self.run_dirs = [Path(path) for path in run_dirs]

    def __len__(self):
        return len(self.run_dirs)

    def __getitem__(self, index):
        run_dir = self.run_dirs[index]

        positions, targets, solid, nx, ny = load_case(run_dir)
        edge_index = build_edges(solid, nx, ny)
        inputs = to_torch(positions, targets, edge_index, nx, ny, "cpu")
        return inputs


if __name__ == "__main__":
    generate_cases(300)