"""Integration checks for saved-run discovery and local gallery serving."""
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from flow_viewer import ViewerHandler, gallery_entries


class GalleryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "flow_viewer.html").write_text("<title>Saved flow</title>", encoding="utf-8")

    def run_folder(self, name, metadata=None):
        folder = self.root / "solver_runs" / name
        folder.mkdir(parents=True)
        (folder / "fields.json").write_text('{"schema":"comms-flow-v1"}', encoding="utf-8")
        if metadata is not None:
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return folder

    def start_server(self):
        class QuietHandler(ViewerHandler):
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, root=self.root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_missing_folder_and_incomplete_run(self):
        self.assertEqual(gallery_entries(self.root), [])
        (self.root / "solver_runs" / "in-progress").mkdir(parents=True)
        self.assertEqual(gallery_entries(self.root), [])

    def test_discovery_handles_legacy_metadata_and_orders_newest_first(self):
        old = self.run_folder("20260101")
        (old / "metadata.json").write_text("{unfinished", encoding="utf-8")
        new = self.run_folder("20260201 #1", {"steps": 12300, "max_steps": 100000, "converged": True})
        (new / "flow.png").write_bytes(b"preview")
        runs = gallery_entries(self.root)
        self.assertEqual([run["id"] for run in runs], ["20260201 #1", "20260101"])
        self.assertEqual(runs[0]["metadata"]["steps"], 12300)
        self.assertEqual(runs[0]["fields_url"], "solver_runs/20260201%20%231/fields.json")
        self.assertEqual(runs[1]["metadata"], {})
        self.assertIsNone(runs[1]["preview_url"])

    def test_api_refresh_discovers_new_runs_and_serves_selected_fields(self):
        url = self.start_server()
        with urlopen(url + "/api/runs") as response:
            self.assertEqual(json.load(response)["runs"], [])
        self.run_folder("run #1", {"steps": 42})
        with urlopen(url + "/api/runs") as response:
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            runs = json.load(response)["runs"]
        self.assertEqual(len(runs), 1)
        with urlopen(url + "/" + runs[0]["fields_url"]) as response:
            self.assertEqual(json.load(response)["schema"], "comms-flow-v1")
        with urlopen(url + "/flow_viewer.html") as response:
            self.assertIn(b"Saved flow", response.read())

    def test_nonfinite_metadata_does_not_break_gallery(self):
        self.run_folder("run", {"steps": float("nan")})
        url = self.start_server()
        with urlopen(url + "/api/runs") as response:
            self.assertIsNone(json.load(response)["runs"][0]["metadata"]["steps"])

    def test_only_viewer_and_saved_fields_are_served(self):
        (self.root / "private.txt").write_text("not served", encoding="utf-8")
        url = self.start_server()
        for path in ("/private.txt", "/solver_runs/", "/solver_runs/../private.txt", "/solver_runs/x/comms.py"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(url + path)
            self.assertEqual(error.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
