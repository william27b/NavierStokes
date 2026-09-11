"""Measure and screen saved D2Q9 runs without changing archives or training."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np


SOURCES = [
    {
        "title": "OpenLB: compressibility effects",
        "url": "https://www.openlb.net/forum/reply/5065/",
        "supports": "Mach-related incompressible-flow error scales as O(Ma^2).",
    },
    {
        "title": "Palabos user guide, chapter 7",
        "url": "https://palabos.unige.ch/files/9515/6509/3036/Palabos_UserGuide.pdf",
        "supports": "Standard BGK uses low Mach numbers to limit compressibility effects.",
    },
]


def _metrics(density, velocity, mask, rho_reference):
    if not mask.any():
        return None
    rho = density[mask]
    speed = np.hypot(velocity[0, mask], velocity[1, mask])
    mach = speed * np.sqrt(3.0)
    deviation = np.abs(rho / rho_reference - 1.0)
    coordinates_yx = np.argwhere(mask)
    return {
        "cells": int(mask.sum()),
        "speed_max": float(speed.max()),
        "mach_max": float(mach.max()),
        "mach_p99": float(np.percentile(mach, 99)),
        "mach_max_at_xy": coordinates_yx[np.argmax(mach)][::-1].tolist(),
        "density_min": float(rho.min()),
        "density_max": float(rho.max()),
        "density_relative_max_abs": float(deviation.max()),
        "density_relative_p99_abs": float(np.percentile(deviation, 99)),
        "density_relative_span": float((rho.max() - rho.min()) / rho_reference),
        "density_max_deviation_at_xy": coordinates_yx[np.argmax(deviation)][::-1].tolist(),
    }


def _inspect_run(run_dir, *, mach_limit, density_limit, rho_reference):
    record = {
        "run_id": run_dir.name, "run_dir": str(run_dir), "seed": None,
        "previously_training_eligible": False, "all_fluid": None,
        "excluding_inlet_outlet": None, "accepted": False, "reasons": [],
    }
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        record.update({
            "seed": metadata.get("generation", {}).get("seed"),
            "converged": bool(metadata.get("converged")),
            "stop_reason": metadata.get("stop_reason"),
            "steps": metadata.get("steps"), "field_stage": metadata.get("field_stage"),
            "kernel_sha256": metadata.get("kernel_sha256"),
            "metadata_invalid_fluid_cells": metadata.get("invalid_fluid_cells"),
            "previously_training_eligible": bool(metadata.get("converged"))
                and metadata.get("invalid_fluid_cells") == 0,
        })
        with np.load(run_dir / "fields.npz", allow_pickle=False) as fields:
            density = fields["rho"].copy()
            velocity = fields["u"].copy()
            pressure = fields["pressure"].copy()
            solid = fields["solid"].astype(bool)
            applied = fields["applied_u"].copy()
        if (solid.ndim != 2 or density.shape != solid.shape or pressure.shape != solid.shape
                or velocity.shape != (2, *solid.shape) or applied.shape[0] != 2):
            raise ValueError("Inconsistent field shapes")
        fluid = ~solid
        record["fluid_cells"] = int(fluid.sum())
        if not fluid.any():
            raise ValueError("No fluid cells")
        if not record["converged"]:
            record["reasons"].append("not_converged")
        if metadata.get("invalid_fluid_cells") != 0:
            record["reasons"].append("metadata_not_zero_invalid_cells")
        finite = np.isfinite(density) & np.isfinite(velocity).all(axis=0) & np.isfinite(pressure)
        record["nonfinite_fluid_cells"] = int((~finite & fluid).sum())
        record["nonpositive_density_cells"] = int((np.isfinite(density) & (density <= 0) & fluid).sum())
        if record["nonfinite_fluid_cells"]:
            record["reasons"].append("nonfinite_fluid_fields")
        if record["nonpositive_density_cells"]:
            record["reasons"].append("nonpositive_density")
        if not np.isfinite(applied).all():
            record["reasons"].append("nonfinite_applied_velocity")
        else:
            record["prescribed_inlet_mach_max"] = float(np.hypot(applied[0], applied[1]).max() * np.sqrt(3.0))
        # A maximum over the finite remainder of a failed run would be misleading.
        # Leave its physical metrics null, and record the invalid-cell counts.
        if not record["nonfinite_fluid_cells"]:
            record["all_fluid"] = _metrics(density, velocity, fluid, rho_reference)
            interior = fluid.copy()
            interior[:, [0, -1]] = False
            record["excluding_inlet_outlet"] = _metrics(density, velocity, interior, rho_reference)
            if record["all_fluid"]["mach_max"] > mach_limit:
                record["reasons"].append("mach_limit_exceeded")
            if record["all_fluid"]["density_relative_max_abs"] > density_limit:
                record["reasons"].append("density_limit_exceeded")
    except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as error:
        record["reasons"].append("archive_unreadable_or_malformed")
        record["error"] = str(error)
    record["accepted"] = not record["reasons"]
    return record


def _quantiles(records, region, metric):
    values = [record[region][metric] for record in records if record[region] is not None]
    if not values:
        return None
    quantiles = np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1])
    return dict(zip(("min", "p25", "median", "p75", "p90", "p95", "p99", "max"), map(float, quantiles)))


def _write_plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = [r for r in report["runs"] if r["previously_training_eligible"] and r["all_fluid"]]
    policy = report["policy"]
    fig, ax = plt.subplots(figsize=(10, 6.8), layout="constrained")
    ax.add_patch(Rectangle((0, 0), policy["mach_limit"], policy["density_relative_limit"] * 100,
                           facecolor="#d4eddb", label="Chosen screening region", zorder=0))
    if rows:
        ax.scatter([r["all_fluid"]["mach_max"] for r in rows],
                   [100 * r["all_fluid"]["density_relative_max_abs"] for r in rows],
                   s=28, alpha=.72, color="#2563a6", edgecolors="white", linewidths=.3,
                   label=f"{len(rows)} currently training-eligible runs")
    ax.axvline(policy["mach_limit"], color="#237744", linestyle="--", linewidth=1.4)
    ax.axhline(policy["density_relative_limit"] * 100, color="#237744", linestyle="--", linewidth=1.4)
    ax.set(xlabel="Maximum local Mach number over fluid cells",
           ylabel="Maximum absolute density deviation from reference (%)",
           title="Saved solver runs versus chosen incompressible-flow screening limits")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)
    ax.legend(loc="upper left", framealpha=.95)
    counts = report["counts"]
    fig.supxlabel(
        f"Limits: Mach <= {policy['mach_limit']:g}, density deviation <= {policy['density_relative_limit']:.0%}. "
        f"Accepted: {counts['accepted']}/{counts['archives']}.\n"
        f"{counts['nonfinite_runs']} runs with nonfinite fields are unplottable. Raw saved boundary cells are included.",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_summary(report, path):
    counts, policy = report["counts"], report["policy"]
    lines = [
        "# Solver-run quality audit", "",
        f"Audited {counts['archives']} saved runs. **{counts['accepted']} pass the chosen screen.**", "",
        f"Existing training filter: {counts['previously_training_eligible']} runs. "
        f"Nonfinite field archives: {counts['nonfinite_runs']}.", "",
        "## Chosen acceptance criteria", "",
        f"- Maximum local Mach number <= {policy['mach_limit']:g}, with Ma = sqrt(3) * hypot(ux, uy).",
        f"- Maximum absolute density deviation <= {policy['density_relative_limit']:.1%}, "
        f"measured as max(abs(rho / {policy['rho_reference']:g} - 1)).",
        "- Converged, finite fluid fields, positive fluid density, and zero invalid cells in metadata.",
        "- Both physical limits apply to every fluid cell, including inlet/outlet columns; solids are excluded.", "",
        policy["rationale"], "", policy["limitations"], "",
        "The reference density is 1 by default because the current kernel imposes rho = 1 at the outlet. "
        "For another solver configuration, supply the matching reference explicitly.", "",
        "## Measured distributions", "",
        "Quantiles below summarize currently training-eligible runs with measurable fields. "
        "Each row summarizes one maximum per run; these are not pooled cell statistics.", "",
        "| Per-run measurement | Minimum | Median | 95th percentile | Maximum |",
        "|---|---:|---:|---:|---:|",
    ]
    for region, label in (("all_fluid", "All fluid"), ("excluding_inlet_outlet", "Without inlet/outlet columns")):
        for metric, suffix, scale in (("mach_max", "maximum Mach", 1),
                                     ("density_relative_max_abs", "maximum density deviation (%)", 100)):
            q = report["distributions"][region][metric]
            if q:
                cells = " | ".join(f"{q[key]*scale:.3f}" for key in ("min", "median", "p95", "max"))
                lines.append(f"| {label}: {suffix} | {cells} |")
    inlet_range = report["prescribed_inlet_mach_range"]
    if inlet_range:
        lines += ["", f"Prescribed inlet Mach range: {inlet_range[0]:.6f} to {inlet_range[1]:.6f}."]
    lines += ["", "## Sensitivity to looser limits", "",
              "These alternatives are diagnostic comparisons, not adopted acceptance criteria.", "",
              "| Maximum Mach allowed | Maximum density deviation allowed | Qualifying runs |",
              "|---:|---:|---:|"]
    for row in report["threshold_sensitivity"]:
        lines.append(f"| {row['mach_limit']:g} | {row['density_relative_limit']:.0%} | {row['accepted']} |")
    legacy_runs = sum(r.get("field_stage") == "after_streaming" for r in report["runs"])
    reconstructed_runs = sum(r.get("field_stage") == "pre_collision_after_boundary_regularization" for r in report["runs"])
    lines += ["", "## Boundary and export stage", "",
              f"{reconstructed_runs} runs store populations and fields after boundary reconstruction and regularization. "
              f"{legacy_runs} runs use the older export of moments before boundary reconstruction.", "",
              "Measurements excluding x=0 and x=nx-1 remain an additional diagnostic; "
              "they do not replace the whole-fluid acceptance check or exclude wall-adjacent cells.", "",
              "No archives, solver settings, training filters, or checkpoints were changed. "
              "The audit does not start training or regenerate cases.", "",
              "## Files", "",
              "- `audit.json`: measurements, per-run reasons, acceptance lists, policy and source links.",
              "- `quality_overview.png`: one point per measurable, currently training-eligible run.", "",
              "## Basis for the screening policy", ""]
    for source in SOURCES:
        lines.append(f"- [{source['title']}]({source['url']}): {source['supports']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_solver_runs(run_root=None, *, mach_limit=.10, density_limit=.05,
                      rho_reference=1.0, output_dir=None):
    """Write a JSON audit, Markdown summary and scatter plot; never filter training.

    Limits are inclusive. Density deviation is relative to the fixed reference,
    not each case's mean. Unreadable/nonfinite runs are retained with fail reasons.
    The defaults are conservative project screening targets, not universal laws.
    """
    if not np.isfinite([mach_limit, density_limit, rho_reference]).all() or min(mach_limit, density_limit, rho_reference) <= 0:
        raise ValueError("Limits and reference density must be finite and positive")
    base = Path(__file__).resolve().parent
    run_root = (base / "solver_runs" if run_root is None else Path(run_root)).resolve()
    if not run_root.is_dir():
        raise ValueError(f"Run directory does not exist: {run_root}")
    run_dirs = sorted({p.parent for pattern in ("*/metadata.json", "*/fields.npz") for p in run_root.glob(pattern)})
    if not run_dirs:
        raise ValueError(f"No solver archives found in {run_root}")
    records = [_inspect_run(path, mach_limit=mach_limit, density_limit=density_limit,
                            rho_reference=rho_reference) for path in run_dirs]
    measurable = [r for r in records if r["previously_training_eligible"] and r["all_fluid"]]
    # These records meet all non-threshold criteria and can be compared under
    # alternate numerical limits without admitting failed or malformed runs.
    valid = [r for r in records if not set(r["reasons"]) - {"mach_limit_exceeded", "density_limit_exceeded"}]
    created = datetime.now(timezone.utc)
    output_dir = (base / "quality_audits" / created.strftime("%Y%m%dT%H%M%S%fZ")
                  if output_dir is None else Path(output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    inlet = [r["prescribed_inlet_mach_max"] for r in measurable if "prescribed_inlet_mach_max" in r]
    report = {
        "created_at": created.isoformat(), "run_root": str(run_root), "output_dir": str(output_dir),
        "policy": {
            "mach_limit": float(mach_limit), "density_relative_limit": float(density_limit),
            "rho_reference": float(rho_reference), "scope": "all_fluid_including_inlet_outlet",
            "mach_formula": "max(hypot(ux, uy)) * sqrt(3)",
            "density_formula": "max(abs(rho / rho_reference - 1))",
            "rationale": "Ma <= 0.10 is a conservative low-Mach target. A separate 5% maximum "
                "density-excursion budget prevents accepting large density changes even at low speed. "
                "These are selected project defaults; the sources motivate low-Mach screening, "
                "not a universal numerical cutoff. Explicit function/CLI limits override the defaults.",
            "limitations": "Passing this screen is not proof of solver accuracy and is not a direct "
                "divergence, conservation, resolution-convergence or boundary-condition test. "
                "O(Ma^2) scaling does not guarantee a particular percentage error. "
                "No thresholds use GNN predictions or loss.",
        },
        "counts": {
            "archives": len(records),
            "previously_training_eligible": sum(r["previously_training_eligible"] for r in records),
            "nonfinite_runs": sum(r.get("nonfinite_fluid_cells", 0) > 0 for r in records),
            "accepted": sum(r["accepted"] for r in records),
            "rejected": sum(not r["accepted"] for r in records),
            "rejection_reasons_overlapping": dict(Counter(reason for r in records for reason in r["reasons"])),
        },
        "distributions": {region: {key: _quantiles(measurable, region, key)
                           for key in ("mach_max", "density_relative_max_abs", "density_relative_span")}
                          for region in ("all_fluid", "excluding_inlet_outlet")},
        "prescribed_inlet_mach_range": [min(inlet), max(inlet)] if inlet else None,
        "threshold_sensitivity": [
            {"mach_limit": ml, "density_relative_limit": dl,
             "accepted": sum(r["all_fluid"]["mach_max"] <= ml
                             and r["all_fluid"]["density_relative_max_abs"] <= dl for r in valid)}
            for ml, dl in dict.fromkeys(((mach_limit, density_limit), (.2, .05), (.3, .05), (.3, .10)))
        ],
        "accepted_run_dirs": [r["run_dir"] for r in records if r["accepted"]],
        "rejected_run_dirs": [r["run_dir"] for r in records if not r["accepted"]],
        "sources": SOURCES, "runs": records,
    }
    (output_dir / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    _write_summary(report, output_dir / "summary.md")
    _write_plot(report, output_dir / "quality_overview.png")
    print(f"Audited {len(records)} runs: {report['counts']['accepted']} accepted, {report['counts']['rejected']} rejected.")
    print(f"Report: {output_dir / 'summary.md'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are not overwritten")
    parser.add_argument("--mach-limit", type=float, default=.10)
    parser.add_argument("--density-limit", type=float, default=.05, help="Fractional maximum deviation from reference density")
    parser.add_argument("--rho-reference", type=float, default=1.0)
    args = parser.parse_args()
    audit_solver_runs(args.run_root, mach_limit=args.mach_limit, density_limit=args.density_limit,
                      rho_reference=args.rho_reference, output_dir=args.output_dir)
