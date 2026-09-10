# D2Q9 solver

Run the simulation from your CuPy environment:

```bash
cd D2Q9
python3 solver.py --solid cylinder
```

Each run creates a new folder named with the local date, time, UTC offset (defaults to Los Angeles), and solid name under `solver_runs/`. The terminal prints its location.

`steps` in `solver.py` is the maximum iteration count. Every `check_every = 100` steps, the script compares the current post-streaming density and both velocity components with the previous check, excluding solids. It stops when the maximum absolute change at any fluid cell is at most `velocity_tolerance = 1e-6` for velocity and `density_tolerance = 1e-6` for density, for `required_stable_checks = 5` consecutive checks. The grid, relaxation time, maximum steps, and inlet components can also be set with --nx, --ny, --tau, --steps, --ux, and --uy. These settings are beside `steps`; the inlet velocity is applied immediately.

The final step is checked even if it is not a multiple of `check_every`. Nonfinite fluid fields stop the run and are reported as a failure to converge. All stop conditions still save the available fields and script. This is a numerical stopping criterion, not a guarantee that a physically unsteady flow will become steady.

Saved `steps` is the actual number completed; `max_steps` records the configured limit. `converged` and `stop_reason` distinguish convergence, reaching the limit, and nonfinite fields. `metadata.json` also records the check settings, number of checks, and last maximum changes. The plot shows the actual step count. Gallery preview cards show completed / maximum steps alongside the outcome (for example, `12,300 / 100,000 steps · converged`). Older runs without `max_steps` show only the completed count.

| File | Contents |
| --- | --- |
| `viewer.html` | Open directly in a browser. Includes this run's data; no server or internet required. |
| `fields.npz` | Full-resolution NumPy arrays, including populations and simulation settings. |
| `fields.json` | Full-resolution pressure, velocity, and solids for the HTML viewer. |
| `solver.py` | Copy of the executed Python with the CUDA kernel, viewer template, geometry functions, and selected configuration embedded. |
| `flow.png` | Pressure and streamline plot. |
| `metadata.json` | Grid size, steps, relaxation time, timestamp, kernel hash, and invalid-cell count. |

The usual `flow.png` beside the main script is also updated.

Start the automatic gallery with Python (no CuPy or extra packages needed):

```bash
python3 flow_viewer.py
```

The launcher opens a local browser page and scans `solver_runs/` beside `flow_viewer.py`, regardless of your working directory. The gallery is one horizontally scrolling row with the active visualizer enlarged in the center and saved-run previews beside it. Scroll sideways (or use the mouse wheel), click a preview, or use the previous/next arrows to center another run. The focused carousel also supports left/right arrow keys. Runs wrap around the active selection in newest-first order, and the newest run loads automatically. Click a card to switch runs. Reloading the page discovers runs saved since it was opened. Only the selected run's full arrays are loaded. The server listens on `127.0.0.1:8000`; use `--port 0` to choose a free port or `--no-browser` to print the URL without opening it. Stop it with Ctrl+C.

The gallery loads its run listing automatically through the local server or hosted site. Each archived `viewer.html` opens directly with its own data already embedded.

The viewer uses a black-and-white interface with a viridis pressure field, black solids, and white particles and trails. Particles spawn with weights from 0.1 at the lowest fluid pressure to 1 at the highest; constant pressure gives uniform spawning. They move through the saved velocity using bilinear interpolation and respawn at walls, domain edges, or the end of their lifetime. Speed and particle-count controls affect only the illustration, and both sliders adjust in increments of 5. Runs are labeled by their shape. Speed and particle count both allow zero: speed zero freezes motion, and particles zero clears particles and trails to show only pressure.

These are **final snapshots**, not a history of every timestep. The particles illustrate a fixed saved field rather than running the fluid solver in the browser. Grid coordinates have x to the right and y upward. Units are lattice units.

Read numerical results in Python:

```python
import numpy as np

run = "solver_runs/YOUR_RUN_FOLDER"
with np.load(f"{run}/fields.npz") as data:
    rho = data["rho"]          # (ny, nx), float64
    pressure = data["pressure"] # rho / 3
    u = data["u"]              # (2, ny, nx): ux, uy
    solid = data["solid"]      # (ny, nx), bool
    f = data["f"]              # (9, ny, nx), final populations
```

The archive also contains `applied_u`, `nx`, `ny`, `tau`, `steps`, `max_steps`, `converged`, and `stop_reason`. JSON fields are flattened in C order: `index = y * nx + x`. They retain the numerical precision of the saved fields. Nonfinite values stay unchanged in NPZ and become `null` in JSON; the viewer excludes those cells and marks them gray.

To rerun an archived experiment, run its `solver.py` using the same CuPy/NumPy/Matplotlib environment. It needs no sibling `kernel.cpp`, `solids.py`, or `flow_viewer.html`. The selected solid and CLI settings become its defaults. A new archive is created under that script's own `solver_runs/` directory.

## Solid functions and batch runs

From the repository root:

```bash
python3 D2Q9/solver.py --solid ellipse
python3 D2Q9/run_solids.py
python3 D2Q9/flow_viewer.py
```

`solids.py` contains independent NumPy functions returning a boolean array with shape
`(ny, nx)`: channel, cylinder, rectangle, ellipse, diamond, two_cylinders,
staggered_cylinders, and constriction. Every function includes the top and bottom
walls required by the current kernel. Register additional functions in `SOLIDS`.
The default cylinder exactly matches the original obstacle.

`run_solids.py` runs each registered geometry once, sequentially. Each gets its own
normal run archive. A separate `solver_runs/batch_TIMESTAMP/` contains a log per
geometry and `summary.json` with the run locations, actual steps, and stop reasons.
Failures are recorded and the remaining geometries are still attempted.

Select a subset or change the shared configuration:

```bash
python3 D2Q9/run_solids.py --solids cylinder diamond --steps 100000 --nx 64 --ny 64
```

No inlet ramp is used. The defaults remain 64 by 64, tau 0.8, inlet velocity
(0.1, 0.0), and at most 100,000 steps per geometry.

Historical archives may still contain a file named `comms.py`; new archives use
`solver.py`. The archive folder is now `solver_runs/`; the solver, local gallery,
saved script paths and batch summaries use that name. The `comms-flow-v1` JSON
schema remains unchanged for compatibility.

Run the viewer integration checks from the repository root with:

```bash
python3 -m unittest discover -s D2Q9 -p 'test_*.py'
```