"""Run solver.py once for each selected solid and save a batch summary."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from solids import SOLIDS


def portable_log(text, root):
    """Remove machine-specific home and project paths from captured output."""
    replacements = ((root.resolve(), "."), (root.resolve().parent, "<project>"),
                    (Path.home().resolve(), "<home>"))
    for path, replacement in replacements:
        for spelling in (str(path), str(path).replace("\\", "/")):
            text = text.replace(spelling, replacement)
    return text


def saved_run_folder(log_text, root):
    """Resolve both current relative log paths and older absolute ones."""
    saved = [line.removeprefix("Saved run: ").strip()
             for line in log_text.splitlines() if line.startswith("Saved run: ")]
    return (root / saved[-1]).resolve() if saved else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solids", nargs="+", choices=tuple(SOLIDS), default=list(SOLIDS))
    for name, kind in (("nx", int), ("ny", int), ("steps", int),
                       ("tau", float), ("ux", float), ("uy", float)):
        parser.add_argument("--" + name, type=kind, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    started_at = datetime.now(timezone.utc)
    batch = root / "solver_runs" / ("batch_" + started_at.strftime("%Y%m%dT%H%M%S.%fZ"))
    batch.mkdir(parents=True, exist_ok=False)
    summary = {"started_at": started_at.isoformat(), "status": "running", "runs": []}
    failed = False
    for name in dict.fromkeys(args.solids):
        command = [sys.executable, str(root / "solver.py"), "--solid", name]
        for option in ("nx", "ny", "steps", "tau", "ux", "uy"):
            value = getattr(args, option)
            if value is not None:
                command.extend(("--" + option, str(value)))
        print(f"Running {name}...", flush=True)
        log_path = batch / (name + ".log")
        with log_path.open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    log.write(portable_log(line, root))
                    log.flush()
                returncode = process.wait()
        row = {"solid_name": name, "returncode": returncode,
               "log": str(log_path.relative_to(root))}
        folder = saved_run_folder(log_path.read_text(encoding="utf-8"), root)
        if folder is not None:
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            row["run_dir"] = str(folder.relative_to(root))
            for key in ("steps", "max_steps", "converged", "stop_reason", "invalid_fluid_cells"):
                row[key] = metadata[key]
        run_failed = returncode != 0 or row.get("stop_reason") == "nonfinite_fields" or folder is None
        failed |= run_failed
        summary["runs"].append(row)
        (batch / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if run_failed:
            print(f"  Failed; see {log_path.relative_to(root).as_posix()}", flush=True)
        else:
            print(f"  {row['stop_reason']}: {row['steps']:,} steps; {row['run_dir']}", flush=True)
    summary["status"] = "completed_with_failures" if failed else "completed"
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    (batch / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Batch summary: {(batch / 'summary.json').relative_to(root).as_posix()}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
