from pathlib import Path

import numpy as np
import torch

from dataset import load_case, build_edges, to_torch
from model import GNN, load_model
from train import train

def _assert_load_close():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(hidden=64, layers=4).to(device)
    model.train()

    run_dir = Path(__file__).resolve().parent.parent
    positions, targets, solid, nx, ny = load_case(run_dir / "D2Q9/solver_runs" / "20260909T192052.196078-0700_cylinder")
    edge_index = build_edges(solid, nx, ny)
    inputs = to_torch(positions, targets, edge_index, nx, ny, device)

    train(model, inputs)

    model.eval()
    with torch.no_grad():
        prediction = model(
            inputs["x"],
            inputs["edge_index"],
            inputs["edge_attr"]
        )

        scale = np.array([0.05, 0.05, 0.0025])
        offset = np.array([0.0, 0.0, 1 / 3])
        predicted_values = prediction.cpu().numpy() * scale + offset

    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": 64,
            "layers": 4,
        },
        Path(__file__).with_name("_test_ckpt.pt"),
    )
    model = load_model(run_dir / "flow_ml/_test_ckpt.pt").to(device)
    model.eval()

    with torch.no_grad():
        loaded_prediction = model(
            inputs["x"],
            inputs["edge_index"],
            inputs["edge_attr"]
        )

        loaded_predicted_values = loaded_prediction.cpu().numpy() * scale + offset

    torch.testing.assert_close(loaded_predicted_values, predicted_values)

if __name__ == "__main__":
    _assert_load_close()