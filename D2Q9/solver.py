from pathlib import Path
import argparse
import ast
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import cupy as cp
import numpy as np
from solids import SOLIDS

# Capture the code and viewer used by this run, before the simulation starts.
script_text = Path(__file__).read_text(encoding="utf-8")
viewer_template = Path(__file__).with_name("flow_viewer.html").read_text(encoding="utf-8")

solids_source = Path(__file__).with_name("solids.py").read_text(encoding="utf-8")

# Load your CUDA source and get its kernel.
source = Path(__file__).with_name("kernel.cpp").read_text(encoding="utf-8")
module = cp.RawModule(code=source)
fused = module.get_function("fused")

default_config = {
    "solid": "cylinder", "nx": 64, "ny": 64, "tau": 0.8,
    "steps": 100_000, "ux": 0.1, "uy": 0.0,
}
parser = argparse.ArgumentParser(description="Run the D2Q9 CUDA solver for one solid geometry.")
parser.add_argument("--solid", choices=tuple(SOLIDS), default=default_config["solid"])
for name, kind in (("nx", int), ("ny", int), ("steps", int),
                   ("tau", float), ("ux", float), ("uy", float)):
    parser.add_argument("--" + name, type=kind, default=default_config[name])
args = parser.parse_args()
nx, ny = args.nx, args.ny
tau = args.tau
steps = args.steps
if nx < 8 or ny < 8:
    parser.error("nx and ny must be at least 8")
if not np.isfinite([tau, args.ux, args.uy]).all() or tau <= 0.5:
    parser.error("tau must be finite and greater than 0.5; inlet velocities must be finite")
solid_name = args.solid

check_every = 100
velocity_tolerance = 1e-6  # Maximum absolute change over a check interval.
density_tolerance = 1e-6   # Pressure = density / 3, so this checks pressure too.
required_stable_checks = 5
rho = cp.ones((ny, nx), dtype=cp.float64)
u = cp.empty((2, ny, nx), dtype=cp.float64)
applied_u = cp.zeros((2, ny, 1), dtype=cp.float64)
applied_u[0, :] = args.ux
applied_u[1, :] = args.uy

c = cp.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
w = cp.array([4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36])
f = w[:, None, None] * rho[None, :, :]
f_out = f.copy()
solid = cp.asarray(SOLIDS[solid_name](nx, ny), dtype=cp.bool_)

block = (16, 16, 1)
grid = (
    (nx + block[0] - 1) // block[0],
    (ny + block[1] - 1) // block[1],
    1,
)
# Compare post-streaming fields so the check uses the same state we save.
fluid = ~solid
if not bool(cp.any(fluid).get()):
    raise ValueError("Convergence checking needs at least one fluid cell.")
if steps < 0 or check_every < 1 or required_stable_checks < 1:
    raise ValueError("steps must be nonnegative; check intervals and counts must be positive.")
if not np.isfinite([velocity_tolerance, density_tolerance]).all() or min(velocity_tolerance, density_tolerance) < 0:
    raise ValueError("Convergence tolerances must be finite and nonnegative.")

previous_rho = f.sum(axis=0)[fluid]
previous_u = (cp.einsum("qa,qyx->ayx", c, f) / f.sum(axis=0)[None, :, :])[:, fluid]
steps_completed = 0
stable_checks = 0
checks_performed = 0
converged = False
stop_reason = "max_steps"
last_velocity_change = None
last_density_change = None

for step in range(steps):
    if (step+1) % 1000 == 0:
        print(f"{solid_name}: step {step+1}; velocity change={last_velocity_change}, density change={last_density_change}", flush=True)

    fused(grid, block, (f, f_out, rho, u, applied_u, solid, np.int32(nx), np.int32(ny), np.float64(tau)))
    f, f_out = f_out, f
    steps_completed = step + 1

    if steps_completed % check_every != 0 and steps_completed != steps:
        continue

    current_rho = f.sum(axis=0)
    current_u = cp.einsum("qa,qyx->ayx", c, f) / current_rho[None, :, :]
    current_rho = current_rho[fluid]
    current_u = current_u[:, fluid]
    checks_performed += 1

    # Transfer only the two maximum changes and a finite-field flag to the CPU.
    velocity_change, density_change, finite_fields = cp.stack((
        cp.max(cp.abs(current_u - previous_u)),
        cp.max(cp.abs(current_rho - previous_rho)),
        cp.all(cp.isfinite(current_u)) & cp.all(cp.isfinite(current_rho)),
    )).get()
    last_velocity_change = float(velocity_change) if np.isfinite(velocity_change) else None
    last_density_change = float(density_change) if np.isfinite(density_change) else None

    if not finite_fields or last_velocity_change is None or last_density_change is None:
        stable_checks = 0
        stop_reason = "nonfinite_fields"
        break

    if velocity_change <= velocity_tolerance and density_change <= density_tolerance:
        stable_checks += 1
    else:
        stable_checks = 0

    previous_rho = current_rho
    previous_u = current_u
    if stable_checks >= required_stable_checks:
        converged = True
        stop_reason = "converged"
        break

if converged:
    print(f"Field converged after {steps_completed:,} steps.", flush=True)
elif stop_reason == "nonfinite_fields":
    print(f"Stopped at step {steps_completed:,}: nonfinite fluid fields detected.", flush=True)
else:
    print(f"Reached the maximum of {steps_completed:,} steps without convergence.", flush=True)

# Recompute fields from the final, post-streaming populations.
rho_final = f.sum(axis=0)
u_final = cp.einsum("qa,qyx->ayx", c, f) / rho_final[None, :, :]

# Save a new archive each time; previous runs are never overwritten.
finished_at = datetime.now(tz=ZoneInfo("America/Los_Angeles"))
run_dir = Path(__file__).resolve().parent / "solver_runs" / (finished_at.strftime("%Y%m%dT%H%M%S.%f%z") + "_" + solid_name)
run_dir.mkdir(parents=True, exist_ok=False)
density_cpu = rho_final.get()
velocity_cpu = u_final.get()
solid_cpu = solid.get()
pressure_cpu = density_cpu / 3.0
valid = np.isfinite(pressure_cpu) & np.isfinite(velocity_cpu).all(axis=0)
metadata = {
    "run_id": run_dir.name,
    "solid_name": solid_name,
    "geometry_sha256": hashlib.sha256(solids_source.encode("utf-8")).hexdigest(),
    "finished_at": finished_at.isoformat(),
    "nx": nx, "ny": ny, "steps": steps_completed, "max_steps": steps, "tau": tau,
    "converged": converged,
    "stop_reason": stop_reason,
    "convergence": {
        "check_every": check_every,
        "velocity_tolerance": velocity_tolerance,
        "density_tolerance": density_tolerance,
        "required_stable_checks": required_stable_checks,
        "stable_checks": stable_checks,
        "checks_performed": checks_performed,
        "last_velocity_change": last_velocity_change,
        "last_density_change": last_density_change,
    },
    "field_stage": "after_streaming",
    "units": "lattice",
    "kernel_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "invalid_fluid_cells": int(np.count_nonzero(~valid & ~solid_cpu)),
}
np.savez_compressed(
    run_dir / "fields.npz",
    f=f.get(), rho=density_cpu, u=velocity_cpu, pressure=pressure_cpu,
    solid=solid_cpu, applied_u=applied_u.get(), solid_name=solid_name,
    nx=nx, ny=ny, tau=tau, steps=steps_completed, max_steps=steps,
    converged=converged, stop_reason=stop_reason,
)
(run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def json_field(values):
    """Keep full resolution; represent nonfinite values as JSON null."""
    flat = values.ravel().astype(object)
    flat[~np.isfinite(values).ravel()] = None
    return flat.tolist()


fields = {
    "schema": "comms-flow-v1", "metadata": metadata, "nx": nx, "ny": ny,
    # C order: index = y * nx + x; y points up.
    "pressure": json_field(pressure_cpu),
    "ux": json_field(velocity_cpu[0]), "uy": json_field(velocity_cpu[1]),
    "solid": solid_cpu.astype(np.uint8).ravel().tolist(),
}
fields_json = json.dumps(fields, separators=(",", ":"), allow_nan=False)
(run_dir / "fields.json").write_text(fields_json, encoding="utf-8")
data_tag = '<script id="run-data" type="application/json">null</script>'
if data_tag not in viewer_template:
    raise ValueError("flow_viewer.html is missing its embedded data placeholder")
viewer = viewer_template.replace(
    data_tag,
    '<script id="run-data" type="application/json">'
    + fields_json.replace("<", "\\u003c") + '</script>',
    1,
)
(run_dir / "viewer.html").write_text(viewer, encoding="utf-8")

# Embed the kernel, viewer, geometry functions, and this run's configuration.
lines = script_text.splitlines(keepends=True)
replacements = {
    "source": source, "viewer_template": viewer_template,
    "solids_source": solids_source, "default_config": vars(args),
}
edits = []
for node in ast.parse(script_text).body:
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in replacements):
        name = node.targets[0].id
        value = replacements[name]
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            literal = '"""' + escaped + '"""'
        else:
            literal = repr(value)
        edits.append((node.lineno, node.end_lineno, f"{name} = {literal}\n"))
    elif isinstance(node, ast.ImportFrom) and node.module == "solids":
        edits.append((node.lineno, node.end_lineno, solids_source + "\n"))
for first, last, replacement in sorted(edits, reverse=True):
    lines[first - 1:last] = [replacement]
(run_dir / "solver.py").write_text("".join(lines), encoding="utf-8")
# Keep saved console output portable and free of local usernames.
run_path = run_dir.relative_to(Path(__file__).resolve().parent)
print(f"Saved run: {run_path.as_posix()}")
print(f"Open in a browser: {(run_path / 'viewer.html').as_posix()}")
if metadata["invalid_fluid_cells"]:
    print(f"Warning: {metadata['invalid_fluid_cells']} fluid cells have nonfinite fields; the viewer excludes them.")

# Plot only after saving the numerical data and reproducible script.
import matplotlib.pyplot as plt

mask = solid_cpu | ~valid
pressure = np.ma.array(pressure_cpu, mask=mask)
ux = np.ma.array(velocity_cpu[0], mask=mask)
uy = np.ma.array(velocity_cpu[1], mask=mask)
y, x = np.indices(pressure.shape)
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("black")
fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
field = ax.imshow(pressure, origin="lower", cmap=cmap, interpolation="nearest")
fig.colorbar(field, ax=ax, label="Pressure (lattice units)")
ax.set(xlabel="x (lattice cells)", ylabel="y (lattice cells)", aspect="equal")
ax.set_title(f"{solid_name.replace('_', ' ')} - pressure and velocity at step {steps_completed:,}", loc="left", pad=14)
ax.streamplot(x, y, ux, uy, color="#daf5e8", density=1.65,
              linewidth=0.45, arrowsize=0.55, maxlength=5, zorder=2)
fig.savefig(run_dir / "flow.png", dpi=180)
fig.savefig(Path(__file__).with_name("flow.png"), dpi=180)
plt.close(fig)

fig, ax = plt.subplots()

for column in (nx // 4, nx // 2, 3 * nx // 4):
    ax.plot(
        ux[:, column],
        np.arange(ny),
        label=f"x = {column}",
    )

ax.set(
    xlabel="Horizontal velocity",
    ylabel="y (lattice cells)",
    title="Velocity across the channel",
)
ax.legend()
ax.grid(alpha=0.3)

fig.savefig(run_dir / "velocity_profiles.png", dpi=180)
plt.close(fig)