from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import cupy as cp
import numpy as np

class D2Q9Solver:
    def __init__(self):
        # Capture the code and viewer used by this run, before the simulation starts.
        # Load your CUDA source and get its kernel.
        kernel_path = Path(__file__).resolve().parent.parent / "D2Q9" / "kernel.cpp"
        self.source = kernel_path.read_text(encoding="utf-8")
        module = cp.RawModule(code=self.source)
        self.fused = module.get_function("fused")

    def solve(
            self, solid, solid_name, *, ux, tau, uy=0.0,
            max_steps=100_000,
            check_every=100,
            velocity_tolerance=1e-6,
            density_tolerance=1e-6,
            required_stable_checks=5,
            extra_metadata=None,
            save=True
    ):  
        print("Running D2Q9 solver...")
        
        if not np.isfinite([ux, uy, tau]).all() or tau <= 0.5:
            raise ValueError("Flow parameters must be finite, and tau must exceed 0.5")
        
        solid = cp.asarray(solid, dtype=cp.bool_)
        ny, nx = solid.shape

        rho = cp.ones((ny, nx), dtype=cp.float64)
        u = cp.empty((2, ny, nx), dtype=cp.float64)
        applied_u = cp.zeros((2, ny, 1), dtype=cp.float64)
        applied_u[0, :] = ux
        applied_u[1, :] = uy

        c = cp.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])
        w = cp.array([4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36])
        f = w[:, None, None] * rho[None, :, :]
        f_out = f.copy()

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
        if max_steps < 0 or check_every < 1 or required_stable_checks < 1:
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

        for step in range(max_steps):
            if (step+1) % 1000 == 0:
                print(f"step {step+1}; velocity change={last_velocity_change}, density change={last_density_change}", flush=True)

            self.fused(grid, block, (f, f_out, rho, u, applied_u, solid, np.int32(nx), np.int32(ny), np.float64(tau)))
            f, f_out = f_out, f
            steps_completed = step + 1

            if steps_completed % check_every != 0 and steps_completed != max_steps:
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
        if save: run_dir.mkdir(parents=True, exist_ok=False)
        density_cpu = rho_final.get()
        velocity_cpu = u_final.get()
        solid_cpu = solid.get()
        pressure_cpu = density_cpu / 3.0
        valid = np.isfinite(pressure_cpu) & np.isfinite(velocity_cpu).all(axis=0)

        metadata = {
            "run_id": run_dir.name,
            "solid_name": solid_name,
            "finished_at": finished_at.isoformat(),
            "nx": nx, "ny": ny, "steps": steps_completed, "max_steps": max_steps, "tau": tau,
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
            "kernel_sha256": hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            "invalid_fluid_cells": int(np.count_nonzero(~valid & ~solid_cpu)),
        } | (extra_metadata or {})

        if save:
            np.savez_compressed(
                run_dir / "fields.npz",
                f=f.get(), rho=density_cpu, u=velocity_cpu, pressure=pressure_cpu,
                solid=solid_cpu, applied_u=applied_u.get(), solid_name=solid_name,
                nx=nx, ny=ny, tau=tau, steps=steps_completed, max_steps=max_steps,
                converged=converged, stop_reason=stop_reason,
            )
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "f": f.get(),
            "rho": density_cpu,
            "u": velocity_cpu,
            "pressure": pressure_cpu,
            "solid": solid_cpu,
            "applied_u": applied_u.get(),
            "solid_name": solid_name,
            "nx": nx,
            "ny": ny,
            "tau": tau,
            "steps": steps_completed,
            "max_steps": max_steps,
            "converged": converged,
            "stop_reason": stop_reason,
            "metadata": metadata
        }