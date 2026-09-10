"""Launch the flow viewer and discover saved runs beside this file."""
import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
from urllib.parse import quote, unquote, urlsplit
import webbrowser


def gallery_entries(root):
    """List lightweight previews without loading the full simulation arrays."""
    runs = (root / "solver_runs").resolve()
    if not runs.is_dir():
        return []
    entries = []
    for folder in sorted(runs.iterdir(), key=lambda path: path.name, reverse=True):
        if not folder.is_dir() or not folder.resolve().is_relative_to(runs):
            continue
        fields = folder / "fields.json"
        if not fields.is_file() or not fields.resolve().is_relative_to(runs):
            continue
        try:
            metadata_path = folder / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"), parse_constant=lambda _: None) if (
                metadata_path.resolve().is_relative_to(runs) and metadata_path.is_file()
            ) else {}
            if not isinstance(metadata, dict):
                metadata = {}
        except (OSError, ValueError):
            metadata = {}
        base = "solver_runs/" + quote(folder.name, safe="") + "/"
        preview = folder / "flow.png"
        entries.append({
            "id": folder.name,
            "metadata": {key: metadata[key] for key in (
                "finished_at", "solid_name", "nx", "ny", "steps", "max_steps", "converged", "stop_reason"
            ) if key in metadata},
            "fields_url": base + "fields.json",
            "preview_url": base + "flow.png" if (
                preview.is_file() and preview.resolve().is_relative_to(runs)
            ) else None,
        })
    return entries


class ViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root.resolve()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.serve(head=False)

    def do_HEAD(self):
        self.serve(head=True)

    def serve(self, head):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/runs":
            try:
                payload = json.dumps({"runs": gallery_entries(self.root)}, allow_nan=False).encode("utf-8")
            except (OSError, ValueError):
                self.send_error(500, "Could not read the saved-run folder")
                return
            self.headers_for("application/json; charset=utf-8", len(payload))
            if not head:
                self.wfile.write(payload)
            return

        if path in ("/", "/flow_viewer.html"):
            target = self.root / "flow_viewer.html"
            content_type = "text/html; charset=utf-8"
        else:
            parts = path.lstrip("/").split("/")
            if (len(parts) != 3 or parts[0] != "solver_runs" or
                    parts[1] in ("", ".", "..") or
                    parts[2] not in ("fields.json", "flow.png")):
                self.send_error(404)
                return
            target = self.root.joinpath(*parts).resolve()
            if not target.is_relative_to((self.root / "solver_runs").resolve()):
                self.send_error(404)
                return
            content_type = "application/json; charset=utf-8" if target.suffix == ".json" else "image/png"

        try:
            with target.open("rb") as source:
                self.headers_for(content_type, target.stat().st_size)
                if not head:
                    shutil.copyfileobj(source, self.wfile)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def headers_for(self, content_type, size):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000, help="Local port (default: 8000; 0 chooses a free port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    handler = partial(ViewerHandler, root=root)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        parser.exit(1, f"Could not start the viewer: {error}. Try --port 0.\n")
    with server:
        url = f"http://127.0.0.1:{server.server_port}/flow_viewer.html"
        print(f"Flow gallery: {url}\nScanning: {root / 'solver_runs'}\nPress Ctrl+C to stop.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
