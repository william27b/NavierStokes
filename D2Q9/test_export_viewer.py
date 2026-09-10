"""Check static export contents, stale-run cleanup and repository-subpath URLs."""
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import unquote
from urllib.request import urlopen

from export_viewer import export_viewer, LOCAL_MANIFEST, STATIC_MANIFEST

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5XcAAAAASUVORK5CYII=")


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "D2Q9"
        self.source = self.project / "solver_runs"
        self.source.mkdir(parents=True)
        self.output = self.root / "NavierStokes"
        (self.project / "flow_viewer.html").write_text("<html><head>" + LOCAL_MANIFEST + "</head></html>")

    def run_folder(self, name="run #1"):
        folder = self.source / name
        folder.mkdir()
        fields = {"schema": "comms-flow-v1", "nx": 2, "ny": 2,
                  "pressure": [0.3] * 4, "ux": [0.1] * 4, "uy": [0.0] * 4,
                  "solid": [0] * 4,
                  "metadata": {"solid_name": "cylinder", "steps": 500, "max_steps": 1000,
                               "converged": True, "private_note": "do not publish"},
                  "private_note": "also exclude this"}
        (folder / "fields.json").write_text(json.dumps(fields))
        (folder / "flow.png").write_bytes(PNG)
        (folder / "solver.py").write_text("# archived source")
        (folder / "run.log").write_text("local diagnostics")
        (folder / "fields.npz").write_bytes(b"not a public asset")
        return folder

    def test_all_runs_and_only_public_assets(self):
        self.run_folder()
        self.run_folder("run #2")
        (self.source / "batch_example").mkdir()
        manifest = export_viewer(self.project, self.output)
        self.assertEqual(len(manifest["runs"]), 2)
        self.assertIn(STATIC_MANIFEST, (self.output / "index.html").read_text())
        self.assertEqual((self.output / "preview.png").read_bytes(), PNG)
        self.assertTrue((self.output / ".nojekyll").exists())
        for run in manifest["runs"]:
            fields = json.loads((self.output / unquote(run["fields_url"])).read_text())
            self.assertNotIn("private_note", fields)
            self.assertNotIn("private_note", fields["metadata"])
            self.assertEqual(set((self.output / "runs" / run["id"]).iterdir()),
                             {self.output / "runs" / run["id"] / name for name in ("fields.json", "flow.png")})

    def test_stale_files_removed_other_documents_preserved(self):
        folder = self.run_folder()
        export_viewer(self.project, self.output)
        (self.output / "GITHUB_PAGES.md").write_text("instructions")
        (folder / "fields.json").unlink()
        manifest = export_viewer(self.project, self.output)
        self.assertEqual(manifest["runs"], [])
        self.assertFalse((self.output / "runs" / folder.name).exists())
        self.assertFalse((self.output / "preview.png").exists())
        self.assertEqual((self.output / "GITHUB_PAGES.md").read_text(), "instructions")

    def test_bad_input_leaves_previous_export_intact(self):
        folder = self.run_folder()
        export_viewer(self.project, self.output)
        before = (self.output / "runs.json").read_bytes()
        data = json.loads((folder / "fields.json").read_text())
        data["pressure"] = ["not a number"] * 4
        (folder / "fields.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            export_viewer(self.project, self.output)
        self.assertEqual((self.output / "runs.json").read_bytes(), before)

    def test_previous_manifest_cannot_delete_outside_export(self):
        self.run_folder()
        export_viewer(self.project, self.output)
        sentinel = self.root / "keep.json"
        sentinel.write_text("keep")
        (self.output / "runs.json").write_text(json.dumps({"runs": [{"fields_url": "../keep.json"}]}))
        with self.assertRaises(ValueError):
            export_viewer(self.project, self.output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_idempotent_export_and_repository_subpath_serving(self):
        self.run_folder()
        export_viewer(self.project, self.output)
        before = {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        export_viewer(self.project, self.output)
        self.assertEqual(before, {p.relative_to(self.output): p.read_bytes() for p in self.output.rglob("*") if p.is_file()})
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(self.root)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}/NavierStokes/"
        with urlopen(base) as response:
            self.assertIn(STATIC_MANIFEST.encode(), response.read())
        with urlopen(base + "runs.json") as response:
            manifest = json.load(response)
        for run in manifest["runs"]:
            with urlopen(base + run["fields_url"]) as response:
                self.assertEqual(json.load(response)["nx"], 2)
            with urlopen(base + run["preview_url"]) as response:
                self.assertEqual(response.read(), PNG)


if __name__ == "__main__":
    unittest.main()
