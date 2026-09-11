# Boundary-aware multiscale flow model

Existing `model.py`, `train.py`, and old checkpoints remain usable. The new experiment has a separate entry point.

From `/home/william/NavierStokes/flow_ml`:

```bash
python test_multiscale.py
python train_multiscale.py --tag multiscale_v1 --epochs 160 --lr .0003
python train_multiscale.py --tag multiscale_v1 --epochs 160 --lr .0003 --resume
python train_multiscale.py --tag multiscale_v1 --final
python report_multiscale.py training_runs/multiscale_v1
```

Resume uses `last.pt` and the existing learning-rate schedule; keep the original experiment settings. `best.pt` is selected on validation only. Test labels are read for model evaluation only with `--final`.

```python
from train_multiscale import load_new
from multiscale_graph import load_graph, to_device
import torch

model, loss_fn, checkpoint = load_new('training_runs/multiscale_v1/best.pt', 'cuda')
graph = to_device(load_graph(run_directory), 'cuda')
model.eval()
with torch.inference_mode():
    normalized_prediction = model(graph)
```

The output columns are `ux/.05`, `uy/.05`, `(p-1/3)/.0025`, matching the existing dataset. For geometry-only inference, call `build_graph(solid, inlet_velocity=(ux,uy), tau=tau)`; targets are not needed by the model.

`multiscale_graph.py` builds/caches geometry and fluid-connected pooling maps. `multiscale_model.py` contains the model and losses. `train_multiscale.py` trains, saves full state, evaluates, and creates separate field plots. `prepare_multiscale_data.py` creates the independently screened validation/test archives in `heldout_runs` without adding them to the training directory.

The present experiment uses five levels, sufficient for 64×64 grids. Increase hierarchy depth with larger grids to retain domain-scale communication, then retrain and validate the larger-resolution problem. The current results do not certify resolution transfer.
