"""Export all saved D2Q9 runs as a static GitHub Pages gallery (standard library only)."""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

META_FIELDS = (
    "finished_at", "solid_name", "steps", "max_steps", "tau", "converged",
    "stop_reason", "field_stage", "units", "invalid_fluid_cells",
)
LOCAL_MANIFEST = '<meta name="flow-runs-url" content="api/runs">'
STATIC_MANIFEST = '<meta name="flow-runs-url" content="runs.json">'


def read_public_fields(path, run_id):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "comms-flow-v1":
        raise ValueError(f"{run_id}: unsupported field schema")
    nx, ny = data.get("nx"), data.get("ny")
    if type(nx) is not int or type(ny) is not int or min(nx, ny) < 2 or nx * ny > 4_000_000:
        raise ValueError(f"{run_id}: invalid or unsupported grid dimensions")
    public = {"schema": data["schema"], "nx": nx, "ny": ny}
    for key in ("pressure", "ux", "uy", "solid"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != nx * ny:
            raise ValueError(f"{run_id}: wrong array length for {key}")
        if key == "solid":
            valid = all(type(v) in (int, bool) and v in (0, 1) for v in values)
        else:
            valid = all(v is None or type(v) in (int, float) and math.isfinite(v) for v in values)
        if not valid:
            raise ValueError(f"{run_id}: invalid values in {key}")
        public[key] = values
    raw_meta = data.get("metadata") or {}
    if not isinstance(raw_meta, dict):
        raise ValueError(f"{run_id}: metadata must be an object")
    metadata = {"run_id": run_id, "nx": nx, "ny": ny}
    for key in META_FIELDS:
        value = raw_meta.get(key)
        if value is None or type(value) is bool or type(value) is int:
            if key in raw_meta:
                metadata[key] = value
        elif type(value) is float and math.isfinite(value):
            metadata[key] = value
        elif isinstance(value, str) and len(value) <= 256:
            metadata[key] = value
    public["metadata"] = metadata
    return public


def managed_path(output, relative):
    """Only allow generated files inside docs/runs/<run>/."""
    path = PurePosixPath(relative)
    if (path.is_absolute() or len(path.parts) != 3 or path.parts[0] != "runs"
            or path.parts[1] in (".", "..") or path.parts[2] not in ("fields.json", "flow.png")):
        raise ValueError("Invalid path in the previous gallery manifest")
    target = output.joinpath(*path.parts)
    if not target.resolve().is_relative_to(output.resolve()):
        raise ValueError("Gallery output must stay inside the export folder")
    return target


def export_viewer(project, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    source = (project / "solver_runs").resolve()
    if not source.is_dir():
        raise ValueError("No solver_runs directory exists beside the solver")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("The public export and local archive folders must be separate")
    template = (project / "flow_viewer.html").read_text(encoding="utf-8")
    if template.count(LOCAL_MANIFEST) != 1:
        raise ValueError("The viewer template is missing its manifest configuration")
    assets, runs = {}, []
    for folder in sorted(source.iterdir(), key=lambda p: p.name, reverse=True):
        if not folder.is_dir():
            continue
        if folder.is_symlink():
            raise ValueError("Symlinked run folders cannot be exported")
        fields = folder / "fields.json"
        if not fields.is_file():
            continue  # Batch logs and incomplete directories have no display data.
        if fields.is_symlink() or not fields.resolve().is_relative_to(source):
            raise ValueError("Field files must be inside the local run folder")
        public = read_public_fields(fields, folder.name)
        base = "runs/" + folder.name + "/"
        url_base = "runs/" + quote(folder.name, safe="") + "/"
        assets[base + "fields.json"] = json.dumps(public, separators=(",", ":"), allow_nan=False).encode("utf-8")
        preview = folder / "flow.png"
        preview_url = None
        if preview.is_file():
            if preview.is_symlink() or not preview.resolve().is_relative_to(source):
                raise ValueError("Preview files must be inside the local run folder")
            image = preview.read_bytes()
            if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"{folder.name}: preview is not a PNG file")
            assets[base + "flow.png"] = image
            preview_url = url_base + "flow.png"
        runs.append({"id": folder.name, "metadata": public["metadata"],
                     "fields_url": url_base + "fields.json", "preview_url": preview_url})

    # Only remove stale files named in our previous manifest; preserve other docs.
    previous = output / "runs.json"
    old_assets = set()
    if previous.is_file():
        old = json.loads(previous.read_text(encoding="utf-8"))
        for run in old.get("runs", []):
            for key in ("fields_url", "preview_url"):
                if run.get(key):
                    relative = unquote(run[key])
                    managed_path(output, relative)
                    old_assets.add(relative)
    for relative in set(assets) | old_assets:
        managed_path(output, relative)
    for name in ("index.html", "runs.json", "preview.png", ".nojekyll"):
        if (output / name).is_symlink():
            raise ValueError("Generated site files cannot be symlinks")

    output.mkdir(parents=True, exist_ok=True)
    for relative, content in assets.items():
        target = managed_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    for relative in old_assets - assets.keys():
        target = managed_path(output, relative)
        if target.is_file():
            target.unlink()
        if target.parent.is_dir() and not any(target.parent.iterdir()):
            target.parent.rmdir()
    thumbnail = next((r for r in runs if r["metadata"].get("solid_name") == "cylinder" and r["preview_url"]), None)
    thumbnail = thumbnail or next((r for r in runs if r["preview_url"]), None)
    if thumbnail:
        (output / "preview.png").write_bytes(assets[unquote(thumbnail["preview_url"])])
    elif (output / "preview.png").is_file():
        (output / "preview.png").unlink()
    (output / "index.html").write_text(template.replace(LOCAL_MANIFEST, STATIC_MANIFEST), encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    manifest = {"schema": "flow-gallery-v1", "runs": runs}
    (output / "runs.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project.parent / "docs",
                        help="Public site folder (default: repository docs/)")
    args = parser.parse_args()
    try:
        manifest = export_viewer(project, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(f"Exported {len(manifest['runs'])} runs. Commit the generated docs/ site to publish it.")
    print("Preview: python3 -m http.server 8000 --directory docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
