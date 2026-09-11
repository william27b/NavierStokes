from pathlib import Path

import numpy as np
import torch

from dataset import load_case, build_edges, to_torch

from model import GNN, load_model
from train import train

from utils import plot

def evaluate_case(model, case_path, case_name, device):
    torch.manual_seed(0)

    positions, targets, solid, nx, ny = load_case(case_path)
    edge_index = build_edges(solid, nx, ny)
    inputs = to_torch(positions, targets, edge_index, nx, ny, device)

    model.eval()
    with torch.no_grad():
        prediction = model(
            inputs["x"],
            inputs["edge_index"],
            inputs["edge_attr"]
        )

        rmse = (prediction - inputs["y"]).square().mean(dim=0).sqrt()
        print(f"Case: {case_name}")
        print(f"Normalized RMSE [ux, uy, p]: {rmse.cpu().tolist()}")

    
        scale = np.array([0.05, 0.05, 0.0025])
        offset = np.array([0.0, 0.0, 1 / 3])
        predicted_values = prediction.cpu().numpy() * scale + offset
        y_value = inputs["y"].cpu().numpy() * scale + offset

        simulated_grid = np.full((int(ny), int(nx), 3), np.nan)
        simulated_grid[~solid] = y_value

        predicted_grid = np.full((int(ny), int(nx), 3), np.nan)
        predicted_grid[~solid] = predicted_values

        error_grid = np.full((int(ny), int(nx), 3), np.nan)
        error_grid[~solid] = predicted_values - y_value

        plot(simulated_grid, solid, nx, ny, title="Simulated", save=f"sim_plot_{case_name}.png")
        plot(predicted_grid, solid, nx, ny, title="Predicted", save=f"hat_plot_{case_name}.png")
        plot(error_grid, solid, nx, ny, title="Error", save=f"err_plot_{case_name}.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(__file__).resolve().parent.parent

    model_path = run_dir / "flow_ml/model.pt"

    case_path = run_dir / "D2Q9/solver_runs" / "20260909T192104.046029-0700_ellipse"
    case_name = "ellipse"

    model = load_model(model_path).to(device)
    model.eval()

    evaluate_case(model, case_path, case_name, device)