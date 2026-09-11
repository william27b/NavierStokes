"""Save pressure/quiver/solid overlays for random or explicitly selected runs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def _read_case(run_dir):
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "fields.npz") as fields:
        pressure = fields["pressure"].copy()
        velocity = fields["u"].copy()
        density = fields["rho"].copy()
        solid = fields["solid"].astype(bool)
        applied_u = fields["applied_u"].copy()
    if pressure.shape != solid.shape or density.shape != solid.shape or velocity.shape != (2, *solid.shape):
        raise ValueError(f"Inconsistent field shapes in {run_dir}")
    fluid = ~solid
    finite = np.isfinite(pressure) & np.isfinite(density) & np.isfinite(velocity).all(axis=0)
    if not fluid.any() or not finite[fluid].all():
        raise ValueError(f"Nonfinite or empty fluid fields in {run_dir}")
    speed = np.linalg.norm(velocity, axis=0)
    stats = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "seed": metadata.get("generation", {}).get("seed"),
        "geometry": metadata.get("solid_name", ""),
        "steps": metadata["steps"],
        "field_stage": metadata.get("field_stage"),
        "pressure_min": float(pressure[fluid].min()),
        "pressure_max": float(pressure[fluid].max()),
        "density_min": float(density[fluid].min()),
        "density_max": float(density[fluid].max()),
        "density_p01": float(np.percentile(density[fluid], 1)),
        "density_p99": float(np.percentile(density[fluid], 99)),
        "speed_max": float(speed[fluid].max()),
        "speed_p95": float(np.percentile(speed[fluid], 95)),
        "mach_max": float(speed[fluid].max() * np.sqrt(3.0)),
        "pressure_density_max_error": float(np.abs(pressure[fluid] - density[fluid] / 3).max()),
    }
    inlet_fluid = fluid[:, 0]
    if inlet_fluid.any():
        inlet_error = velocity[:, inlet_fluid, 0] - applied_u[:, inlet_fluid, 0]
        stats["saved_inlet_velocity_max_error"] = float(np.linalg.norm(inlet_error, axis=0).max())
    return pressure, velocity, solid, stats


def _plot_case(case, output_path, *, index, count, stride, arrow_scale, reference_velocity, pressure_limits,
               selection_label="Random training run"):
    pressure, velocity, solid, stats = case
    ny, nx = solid.shape
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#20252b")
    fig, ax = plt.subplots(figsize=(10.5, 8.4), layout="constrained")
    limits = {} if pressure_limits is None else dict(zip(("vmin", "vmax"), pressure_limits))
    field = ax.imshow(
        np.ma.array(pressure, mask=solid), origin="lower", cmap=cmap,
        interpolation="nearest", **limits,
    )
    fig.colorbar(field, ax=ax, label="Pressure p (lattice units)", fraction=0.045, pad=0.04)

    # Include the fluid rows next to both walls and the outlet column.
    xs = np.unique(np.r_[np.arange(0, nx, stride), nx - 1])
    ys = np.unique(np.r_[np.arange(1, ny - 1, stride), ny - 2])
    x, y = np.meshgrid(xs, ys)
    mask = solid[y, x]
    quiver = ax.quiver(
        x, y,
        np.ma.array(velocity[0, y, x], mask=mask),
        np.ma.array(velocity[1, y, x], mask=mask),
        angles="xy", scale_units="xy", scale=arrow_scale, pivot="mid",
        color="#f8fafc", edgecolor="#172b3a", linewidth=0.35,
        width=0.003, headwidth=3.4, headlength=4.3, minlength=0, zorder=3,
    )
    # Draw the actual pixel mask above arrows so glyphs do not cover solids.
    solid_cmap = ListedColormap(["#20252b"])
    solid_cmap.set_bad((0, 0, 0, 0))
    ax.imshow(
        np.ma.masked_where(~solid, np.ones_like(solid)),
        origin="lower", interpolation="nearest", cmap=solid_cmap,
        vmin=0, vmax=1, zorder=4,
    )
    ax.legend(
        handles=[Patch(facecolor="#20252b", label="Solid")],
        loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
        borderaxespad=0, fontsize=10,
    )
    ax.quiverkey(
        quiver, X=0.59, Y=1.03, U=reference_velocity,
        label=f"{reference_velocity:g} lattice velocity",
        labelpos="E", coordinates="axes", fontproperties={"size": 10},
    )
    seed = stats["seed"] if stats["seed"] is not None else "unrecorded"
    ax.set_title(
        f"{selection_label} {index}/{count} | seed {seed} | {stats['steps']:,} steps\n"
        f"{stats['geometry']}", loc="left", fontsize=12, pad=42,
    )
    ax.set(
        xlabel="x (lattice cells)", ylabel="y (lattice cells)", aspect="equal",
        xlim=(-0.5, nx - 0.5), ylim=(-0.5, ny - 0.5),
    )
    fig.supxlabel(
        f"Fluid density: {stats['density_min']:.3f} to {stats['density_max']:.3f}   |   "
        f"Max speed: {stats['speed_max']:.3f}   |   Max Mach: {stats['mach_max']:.3f}\n"
        "Raw archived fields; uniform arrow scale across this selection.",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def visualize_solver_runs(run_root=None, *, count=5, seed=42, output_dir=None, stride=3,
                          common_pressure_scale=False, case_seeds=None):
    """Select converged training-eligible runs and save one PNG each.

    Supply case_seeds to plot those generation seeds in the given order;
    count and seed only control random selection when case_seeds is None.
    Pressure is plotted in raw lattice units. Arrow lengths share one scale;
    each pressure colorbar uses its full case range unless common_pressure_scale
    is true. selection.json records the selection, paths, and field diagnostics.
    This reads existing archives and does not run the solver or change training.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    base = Path(__file__).resolve().parent
    run_root = (base / "solver_runs" if run_root is None else Path(run_root)).resolve()
    eligible = []
    for metadata_path in sorted(run_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("converged") and metadata.get("invalid_fluid_cells") == 0
                and (metadata_path.parent / "fields.npz").is_file()):
            eligible.append(metadata_path.parent)
    if case_seeds is None:
        if len(eligible) < count:
            raise ValueError(f"Requested {count} runs, but only {len(eligible)} eligible runs exist in {run_root}")
        indices = np.random.default_rng(seed).choice(len(eligible), size=count, replace=False)
        selected = [eligible[int(index)] for index in indices]
        selection_label = "Random training run"
    else:
        case_seeds = list(case_seeds)
        if not case_seeds or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 0
            for value in case_seeds
        ):
            raise ValueError("case_seeds must contain nonnegative integer generation seeds")
        case_seeds = [int(value) for value in case_seeds]
        if len(set(case_seeds)) != len(case_seeds):
            raise ValueError("case_seeds must not contain duplicates")
        by_seed = {}
        for run_dir in eligible:
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            generation_seed = metadata.get("generation", {}).get("seed")
            by_seed.setdefault(generation_seed, []).append(run_dir)
        selected = []
        for generation_seed in case_seeds:
            matches = by_seed.get(generation_seed, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Seed {generation_seed} has {len(matches)} eligible runs in {run_root}; "
                    "exactly one is required"
                )
            selected.append(matches[0])
        count = len(selected)
        selection_label = "Selected training run"
    cases = [_read_case(run_dir) for run_dir in selected]
    reference_speed = max(case[3]["speed_p95"] for case in cases)
    # Adapt to the actual velocities while sharing one scale across all plots.
    reference_speed = reference_speed if reference_speed > 0 else 1.0
    reference_velocity = float(10.0 ** np.floor(np.log10(reference_speed)))
    arrow_scale = reference_speed / (0.8 * stride)
    pressure_limits = None
    if common_pressure_scale:
        pressure_limits = (
            min(case[3]["pressure_min"] for case in cases),
            max(case[3]["pressure_max"] for case in cases),
        )
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        suffix = f"seed{seed}" if case_seeds is None else "selected"
        output_dir = base / "solver_run_plots" / f"{stamp}_{suffix}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(cases, 1):
        case_seed = case[3]["seed"]
        output_path = output_dir / f"run_{index:02d}_seed_{case_seed}.png"
        _plot_case(case, output_path, index=index, count=count, stride=stride,
                   arrow_scale=arrow_scale, reference_velocity=reference_velocity,
                   pressure_limits=pressure_limits,
                   selection_label=selection_label)
        records.append({**case[3], "image_path": str(output_path)})
        print(f"Saved {output_path}")
    report = {
        "selection_mode": "random" if case_seeds is None else "case_seeds",
        "selection_seed": seed if case_seeds is None else None,
        "case_seeds": case_seeds, "eligible_runs": len(eligible),
        "run_root": str(run_root), "output_dir": str(output_dir),
        "arrow_stride": stride, "arrow_scale": arrow_scale,
        "arrow_reference_velocity": reference_velocity,
        "common_pressure_scale": common_pressure_scale,
        "runs": records,
    }
    (output_dir / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case-seeds", type=int, nargs="+",
                        help="Generation seeds to plot in order; overrides random --count/--seed")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--common-pressure-scale", action="store_true")
    args = parser.parse_args()
    visualize_solver_runs(args.run_root, count=args.count, seed=args.seed,
                          output_dir=args.output_dir, stride=args.stride,
                          common_pressure_scale=args.common_pressure_scale,
                          case_seeds=args.case_seeds)
