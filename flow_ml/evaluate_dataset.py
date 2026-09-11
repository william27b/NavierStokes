"""Evaluate a frozen flow GNN, constant baselines, and separate field plots."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np
import torch

from dataset import load_case, build_edges, to_torch
from model import GNN


# Must match dataset.load_case. Every archive is checked against this transform.
SCALE = np.array([0.05, 0.05, 0.0025])
OFFSET = np.array([0.0, 0.0, 1.0 / 3.0])
CHANNELS = ("ux", "uy", "pressure")
METHODS = ("model", "inlet_constant", "label_fitted_constant")


def channel_mse(prediction, target):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 3 or not len(target):
        raise ValueError("Prediction and target must have matching nonempty (nodes, 3) shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite prediction or target")
    return np.square(prediction - target).mean(axis=0)


def metric_summary(case_mses):
    values = np.asarray(case_mses, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite per-case channel MSEs")
    mse = values.mean(axis=0)
    return {
        "normalized_mse": mse.tolist(),
        "normalized_rmse": np.sqrt(mse).tolist(),
        "lattice_rmse": (np.sqrt(mse) * SCALE).tolist(),
        "loss": float(mse.mean()),
    }


def fit_constant(case_targets):
    """Minimizer of case-averaged MSE; a label-fitted diagnostic reference."""
    return np.mean([np.asarray(target, dtype=float).mean(axis=0) for target in case_targets], axis=0)


def reconstruct_grid(values, positions, solid):
    """Map node values by explicit (x, y) coordinates, leaving solids as NaN."""
    values, positions, solid = np.asarray(values), np.asarray(positions), np.asarray(solid, dtype=bool)
    if values.shape != (int((~solid).sum()), 3) or positions.shape != (len(values), 2):
        raise ValueError("Node count does not match the fluid mask")
    coords = positions.astype(np.int64)
    if not np.array_equal(coords, positions):
        raise ValueError("Expected integer lattice coordinates")
    x, y = coords.T
    if np.any(x < 0) or np.any(x >= solid.shape[1]) or np.any(y < 0) or np.any(y >= solid.shape[0]):
        raise ValueError("Coordinate outside the lattice")
    if solid[y, x].any() or len(np.unique(y * solid.shape[1] + x)) != len(values):
        raise ValueError("Nodes must cover each fluid cell exactly once")
    grid = np.full((*solid.shape, 3), np.nan)
    grid[y, x] = values
    return grid


def frozen_predict(model, inputs):
    model.eval()
    with torch.inference_mode():
        result = model(inputs["x"], inputs["edge_index"], inputs["edge_attr"])
    if result.shape != inputs["y"].shape or not torch.isfinite(result).all():
        raise ValueError("Model returned invalid fields")
    return result.detach().cpu().numpy()


def eligible_runs(run_root):
    records = []
    for path in sorted(Path(run_root).glob("*/metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("converged") and metadata.get("invalid_fluid_cells") == 0:
            if not (path.parent / "fields.npz").is_file():
                raise ValueError(f"Missing fields in {path.parent}")
            records.append((path.parent, metadata))
    if not records:
        raise ValueError("No training-eligible runs found")
    return records


def plot_fields(grid, solid, path, *, title, pressure_limits, arrow_scale,
                reference_velocity, residual=False, footer=""):
    fig, ax = plt.subplots(figsize=(10.5, 8.4), layout="constrained")
    cmap = plt.get_cmap("RdBu_r" if residual else "viridis").copy()
    cmap.set_bad("#20252b")
    options = ({"norm": TwoSlopeNorm(vmin=pressure_limits[0], vcenter=0, vmax=pressure_limits[1])}
               if residual else {"vmin": pressure_limits[0], "vmax": pressure_limits[1]})
    field = ax.imshow(np.ma.array(grid[:, :, 2], mask=solid), origin="lower",
                      interpolation="nearest", cmap=cmap, **options)
    label = "Pressure error: prediction - solver" if residual else "Pressure p"
    fig.colorbar(field, ax=ax, fraction=.045, pad=.04, label=label + " (lattice units)")
    ny, nx = solid.shape
    xs = np.unique(np.r_[np.arange(0, nx, 3), nx - 1])
    ys = np.unique(np.r_[np.arange(1, ny - 1, 3), ny - 2])
    x, y = np.meshgrid(xs, ys)
    arrows = ax.quiver(x, y, np.ma.array(grid[y, x, 0], mask=solid[y, x]),
                      np.ma.array(grid[y, x, 1], mask=solid[y, x]),
                      angles="xy", scale_units="xy", scale=arrow_scale, pivot="mid",
                      color="#f8fafc", edgecolor="#172b3a", linewidth=.35,
                      width=.003, headwidth=3.4, headlength=4.3, minlength=0, zorder=3)
    solid_cmap = ListedColormap(["#20252b"])
    solid_cmap.set_bad((0, 0, 0, 0))
    ax.imshow(np.ma.masked_where(~solid, np.ones_like(solid)), origin="lower",
              cmap=solid_cmap, interpolation="nearest", vmin=0, vmax=1, zorder=4)
    arrow_label = "velocity error" if residual else "velocity"
    ax.quiverkey(arrows, X=.57, Y=1.025, U=reference_velocity,
                 label=f"{reference_velocity:g} lattice {arrow_label}", labelpos="E", coordinates="axes")
    ax.set_title(title, loc="left", fontsize=12, pad=42)
    ax.set(xlabel="x (lattice cells)", ylabel="y (lattice cells)", aspect="equal",
           xlim=(-.5, nx-.5), ylim=(-.5, ny-.5))
    fig.supxlabel(footer, fontsize=10)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_plots(selected, output_dir):
    # Target and prediction share all scales across this selection. Residuals
    # share separate symmetric pressure limits and a separate velocity scale.
    pressures, speeds, pressure_errors, velocity_errors = [], [], [], []
    for case in selected:
        for key in ("target", "prediction"):
            pressures.append(case[key][:, 2])
            speeds.append(np.linalg.norm(case[key][:, :2], axis=1))
        error = case["prediction"] - case["target"]
        pressure_errors.append(error[:, 2])
        velocity_errors.append(np.linalg.norm(error[:, :2], axis=1))
    p_limits = (min(v.min() for v in pressures), max(v.max() for v in pressures))
    if p_limits[0] == p_limits[1]:
        p_limits = (p_limits[0]-1e-6, p_limits[1]+1e-6)
    p_error = max(max(np.abs(v).max() for v in pressure_errors), 1e-12)
    speed = max(max(np.percentile(v, 95) for v in speeds), 1e-12)
    error_speed = max(max(np.percentile(v, 95) for v in velocity_errors), 1e-12)
    scales = {"pressure_limits": list(map(float, p_limits)), "pressure_error_limit": float(p_error),
              "velocity_arrow_scale": float(speed/2.4), "error_arrow_scale": float(error_speed/2.4)}
    for case in selected:
        prefix = f"case_{case['index']:03d}_seed_{case['seed']}"
        title = f"Seed {case['seed']} | {case['geometry']}"
        rmse = np.sqrt(case["model_mse"]) * SCALE
        footer = (f"Model RMSE: ux={rmse[0]:.4g}, uy={rmse[1]:.4g}, p={rmse[2]:.4g} (lattice units)\n")
        images = {}
        for kind in ("target", "prediction", "error"):
            residual = kind == "error"
            values = case["prediction"]-case["target"] if residual else case[kind]
            grid = reconstruct_grid(values, case["positions"], case["solid"])
            image_path = output_dir / f"{prefix}_{kind}.png"
            titles = {"target": "Solver target", "prediction": "GNN prediction", "error": "Signed error: GNN - solver"}
            selected_speed = error_speed if residual else speed
            plot_fields(grid, case["solid"], image_path, title=f"{titles[kind]}\n{title}",
                        pressure_limits=(-p_error, p_error) if residual else p_limits,
                        arrow_scale=selected_speed/2.4,
                        reference_velocity=float(10.0**np.floor(np.log10(selected_speed))),
                        residual=residual, footer=footer + (
                            "Arrows show velocity error; symmetric pressure colors. Dark = solid."
                            if residual else "Shared target/prediction scales across the selected cases. Dark = solid."))
            images[kind] = image_path.name
        case["images"] = images
    return scales


def write_metrics_plot(report, output_dir):
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    labels = ("GNN", "Uniform inlet / outlet pressure", "Label-fitted constant")
    colors = ("#2563eb", "#94a3b8", "#f59e0b")
    for index, (method, label, color) in enumerate(zip(METHODS, labels, colors)):
        ax.bar(x + (index-1)*.25, report["metrics"][method]["normalized_mse"], .24, label=label, color=color)
    ax.set_xticks(x, CHANNELS)
    ax.set(ylabel="Case-averaged normalized MSE (lower is better)",
           title=f"Frozen checkpoint versus constants | {report['case_count']} training cases")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.savefig(output_dir/"metrics_comparison.png", dpi=160)
    plt.close(fig)


def write_report(report, output_dir):
    lines = ["# Frozen-checkpoint training evaluation", "",
             f"Checkpoint: `{report['checkpoint']}`", "",
             f"Evaluated **{report['case_count']} cases** without parameter updates. These are training-fit measurements, not validation results.", "",
             "Each case gets equal weight: average squared error over its fluid nodes, then average cases. Solids are excluded; fluid inlet/outlet cells are included. Channel RMSE is the square root of this averaged MSE, not the average of case RMSEs.", "",
             "Normalization: ux/0.05, uy/0.05, (p - 1/3)/0.0025, unchanged from training.", "",
             "| Method | Total normalized MSE | ux MSE | uy MSE | Pressure MSE |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        m = report["metrics"][method]
        lines.append(f"| {method} | {m['loss']:.8g} | " + " | ".join(f"{v:.8g}" for v in m["normalized_mse"]) + " |")
    lines += ["", "| Method | ux RMSE (lattice) | uy RMSE (lattice) | Pressure RMSE (lattice) |", "|---|---:|---:|---:|"]
    for method in METHODS:
        lines.append(f"| {method} | " + " | ".join(f"{v:.8g}" for v in report["metrics"][method]["lattice_rmse"]) + " |")
    lines += ["", "The inlet constant uses each archive's prescribed inlet velocity, uniformly over fluid nodes, and p=1/3. It uses no target labels. The label-fitted constant is one three-component vector fitted across this evaluated dataset using equal case weights; it is an in-sample diagnostic reference, not a held-out result.", "",
              f"Label-fitted constant [ux, uy, p] in lattice units: `{report['label_fitted_constant_lattice']}`.", "",
              f"Model beats the inlet constant on {report['case_wins']['inlet_constant']}/{report['case_count']} cases and the label-fitted constant on {report['case_wins']['label_fitted_constant']}/{report['case_count']} cases (total normalized MSE).", "",
              "The reported epoch loss averages predictions made while parameters change. This report uses only the final checkpoint, so its loss need not equal the epoch-40 log.", "",
              "![Channel comparison](metrics_comparison.png)", "", "## Reproducibly random cases", "",
              "Target and prediction plots share pressure and velocity scales across the selection. Signed-error plots use their own shared scales, with pressure centered at zero. Error arrows represent prediction minus target, not a flow solution.", "",
              "| Seed | Solver target | GNN prediction | Signed error |", "|---|---|---|---|"]
    for case in report["selected_cases"]:
        images = case["images"]
        lines.append(f"| {case['seed']} | [Target]({images['target']}) | [Prediction]({images['prediction']}) | [Error]({images['error']}) |")
    lines += ["", "## Verification", "", "- Every graph target was checked against its raw archived velocity and pressure after normalization.",
              "- Selected graph fields were mapped back using explicit coordinates; solids remain masked.",
              "- Model state tensors and checkpoint SHA-256 were identical before and after evaluation.",
              "- Per-case metrics, checkpoint identity, architecture, selection, and scaling details are in `metrics.json`.", ""]
    (output_dir/"summary.md").write_text("\n".join(lines))


def evaluate_dataset(checkpoint=None, run_root=None, *, output_dir=None, count=5, seed=42, device=None):
    base = Path(__file__).resolve().parent
    checkpoint = (base/"model_v4.pt" if checkpoint is None else Path(checkpoint)).resolve()
    run_root = (base/"solver_runs" if run_root is None else Path(run_root)).resolve()
    runs = eligible_runs(run_root)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(runs):
        raise ValueError("Plot count must be between one and the number of eligible cases")
    selected_indices = list(map(int, np.random.default_rng(seed).choice(len(runs), size=count, replace=False)))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    memory = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = {"hidden": memory["hidden"], "layers": memory["layers"], "global_context": memory.get("global_context", False)}
    model = GNN(**config).to(device)
    model.load_state_dict(memory["model_state"], strict=True)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    targets_by_case, records, selected_by_index = [], [], {}
    for index, (run_dir, metadata) in enumerate(runs):
        positions, targets, solid, nx, ny = load_case(run_dir)
        with np.load(run_dir/"fields.npz") as archive:
            raw = np.stack((archive["u"][0], archive["u"][1], archive["pressure"]), axis=-1)[~solid]
            np.testing.assert_allclose(targets, (raw-OFFSET)/SCALE, rtol=2e-6, atol=2e-7)
            if not (~solid[:, 0]).any():
                raise ValueError(f"No fluid inlet in {run_dir}")
            inlet = archive["applied_u"][:, ~solid[:, 0], 0].mean(axis=1)
        inputs = to_torch(positions, targets, build_edges(solid, nx, ny), nx, ny, device)
        prediction = frozen_predict(model, inputs)
        target = inputs["y"].cpu().numpy().astype(float)
        targets_by_case.append(target)
        constant = (np.array([inlet[0], inlet[1], OFFSET[2]])-OFFSET)/SCALE
        model_mse = channel_mse(prediction, target)
        baseline_mse = channel_mse(np.broadcast_to(constant, target.shape), target)
        record = {"run_dir": str(run_dir), "seed": metadata.get("generation", {}).get("seed"),
                  "nodes": len(target), "model_mse": model_mse.tolist(),
                  "inlet_constant_mse": baseline_mse.tolist(), "model_loss": float(model_mse.mean()),
                  "inlet_constant_lattice": (constant*SCALE+OFFSET).tolist()}
        records.append(record)
        if index in selected_indices:
            selected_by_index[index] = {"index": index, "seed": record["seed"],
                "geometry": metadata.get("solid_name", ""), "model_mse": model_mse,
                "positions": positions, "solid": solid, "target": target*SCALE+OFFSET,
                "prediction": prediction.astype(float)*SCALE+OFFSET}
        if (index+1) % 50 == 0:
            print(f"Evaluated {index+1}/{len(runs)} cases", flush=True)
    fitted = fit_constant(targets_by_case)
    for record, target in zip(records, targets_by_case):
        record["label_fitted_constant_mse"] = channel_mse(np.broadcast_to(fitted, target.shape), target).tolist()
    for key, value in model.state_dict().items():
        if not torch.equal(before[key], value.detach().cpu()):
            raise AssertionError(f"Evaluation changed model state: {key}")
    if checkpoint_hash != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise AssertionError("Checkpoint changed during evaluation")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (base/"evaluations"/stamp if output_dir is None else Path(output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = [selected_by_index[index] for index in selected_indices]
    plot_scales = write_plots(selected, output_dir)
    metrics = {method: metric_summary([record[f"{method}_mse"] for record in records]) for method in METHODS}
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "model_config": config,
              "created_at": datetime.now(timezone.utc).isoformat(), "device": str(device),
              "run_root": str(run_root), "output_dir": str(output_dir), "case_count": len(runs),
              "evaluation_kind": "training_fit", "channels": list(CHANNELS),
              "normalization_scale": SCALE.tolist(), "normalization_offset": OFFSET.tolist(),
              "weighting": "equal cases; equal fluid nodes within each case; equal channels in total loss",
              "metrics": metrics, "label_fitted_constant_normalized": fitted.tolist(),
              "label_fitted_constant_lattice": (fitted*SCALE+OFFSET).tolist(),
              "case_wins": {method: sum(record["model_loss"] < np.mean(record[f"{method}_mse"]) for record in records)
                            for method in METHODS[1:]},
              "selection_seed": seed, "plot_scales": plot_scales,
              "selected_cases": [{"seed": case["seed"], "index": case["index"], "images": case["images"]} for case in selected],
              "runs": records, "checkpoint_unchanged": True, "model_state_unchanged": True}
    # Python bool conversion avoids NumPy integer scalars in the JSON report.
    report["case_wins"] = {key: int(value) for key, value in report["case_wins"].items()}
    (output_dir/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    write_metrics_plot(report, output_dir)
    write_report(report, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics, "case_wins": report["case_wins"]}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    evaluate_dataset(args.checkpoint, args.run_root, output_dir=args.output_dir, count=args.count,
                     seed=args.seed, device=args.device)
