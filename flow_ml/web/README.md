# Flow Canvas

A static, interactive drawing page for the trained multiscale flow GNN. Paint or erase solid cells and see predicted pressure colors and velocity arrows. Inference runs in a Web Worker on the visitor's device, using WebGPU when available and WebAssembly on the CPU otherwise.

## Publish on GitHub Pages

1. Copy **all the contents of this folder**, including `vendor/`, `model.onnx`, `model.json`, and `.nojekyll`, into your repository's Pages publishing folder. You can also put them in a subfolder such as `flow/`.
2. Publish that folder with your normal GitHub Pages workflow.
3. Open its HTTPS URL, for example `https://YOUR-NAME.github.io/flow/`.

No Python service, API key, CDN, special response headers, or build step is required. The HTML needs its adjacent assets; it is not a self-contained single file. The model is about 2.8 MB and the runtime is about 25.8 MB before HTTP compression, downloaded on first load and subject to normal browser caching. Keep the vendored runtime files together at the same version.

For a local preview, serve this folder over HTTP, for example `python3 -m http.server 8000`, then open `http://localhost:8000/`. Double-clicking `index.html` will not allow the browser to fetch its model assets. HTTPS on GitHub Pages or localhost enables WebGPU; the fallback still depends on a modern browser with WebAssembly SIMD support.

## Controls

- Draw solid / Erase, brush diameter, undo / redo. `B` and `E` switch tools; `Ctrl/Cmd+Z` undoes; `Ctrl/Cmd+Shift+Z` redoes. Right-drag erases, and Shift-drag makes a rectangle.
- Circle, Two obstacles, and Clear obstacles provide starting points.
- Save mask / Load mask round-trip JSON with a 64 by 64 boolean array, indexed `solid[y][x]`, with the origin at the bottom left. Boundary cells are restored to the trained channel configuration on import.
- The pressure color scale shows `p - p_out` in lattice units. Full range is the default. Interior contrast clips the color range at the 1st and 99th percentiles; it does not change predictions.
- Toggle velocity arrows or the cell grid; adjust arrow spacing and size; lock the pressure scale for comparisons. Hover for actual lattice pressure and velocity components.
- Save field PNG exports the current field, solids, arrows, axes, and legend.

## Model

The exported weights are the existing `multiscale_v1/best.pt` checkpoint at epoch 150, with five fluid-connected graph levels. No retraining or weight quantization was used. The graph is rebuilt after geometry changes, preserving fluid connectivity during coarsening. Predictions are throttled while drawing, and obsolete results are discarded.

Fixed settings match training: `nx=ny=64`, inlet `ux=0.01`, `uy=0`, `tau=0.53`, and outlet `p=1/3`. Upper and lower walls are fixed; inlet and outlet stay open. This page predicts the model's steady fields; it does not evolve a fluid simulation. Disconnected fluid regions are rejected. Unfamiliar drawings can be less accurate than the held-out test cases.

The checkpoint and ONNX SHA-256 hashes are recorded in `model.json`. Browser outputs use the same target normalization as training: `[ux/0.05, uy/0.05, (p-1/3)/0.0025]`, then convert back to lattice units for display.

## Verification

Open `verify.html` and click each backend button to compare the actual browser worker against stored PyTorch predictions for test cases 3000, 3001, and an empty channel. The page reports the backend actually used, so a GPU test that falls back to CPU is visible. Add `?backend=wasm` to the drawing page's URL to force the CPU fallback.

On the development computer, both backends passed with maximum normalized component error below `1e-6`. Exact graph input parity was also checked against Python, including topology checks for barriers and enclosed pockets. Typical checked WebGPU updates were tens of milliseconds and CPU updates roughly a tenth of a second on that machine; these timings are not a guarantee for other hardware.

## Runtime attribution

The vendored [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) version is 1.29.0, from the official npm package. Its MIT license and third-party notices are included in `vendor/`. The worker follows the runtime's [WebGPU execution](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html) and [WebAssembly environment settings](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html). Single-threaded WASM avoids requiring cross-origin isolation headers on the host.
