from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import load_case, build_edges, to_torch
from model import GNN

from utils import plot

def train(model, inputs, steps=1_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for i in range(steps+1):
        optimizer.zero_grad(set_to_none=True)

        prediction = model(
            inputs["x"],
            inputs["edge_index"],
            inputs["edge_attr"],
        )

        loss = (prediction - inputs["y"]).square().mean()
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(f"Loss: {loss.item()}")

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(hidden=64, layers=4).to(device)
    model.train()

    run_dir = Path(__file__).resolve().parent.parent / "D2Q9/solver_runs" / "20260909T192052.196078-0700_cylinder"
    positions, targets, solid, nx, ny = load_case(run_dir)

    # print(positions.shape, positions.dtype)
    # print(targets.shape, targets.dtype)
    # print(solid.shape, solid.dtype)

    # _validate_build_edges()
    edge_index = build_edges(solid, nx, ny)
    # print(edge_index.shape, edge_index.dtype)

    inputs = to_torch(positions, targets, edge_index, nx, ny, device)

    train(model, inputs)

    model.eval()
    with torch.no_grad():
        prediction = model(
            inputs["x"],
            inputs["edge_index"],
            inputs["edge_attr"]
        )

        rmse = (prediction - inputs["y"]).square().mean(dim=0).sqrt()
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

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(27, 9))

        plot(simulated_grid, solid, nx, ny, title="Simulated", save="sim_plot.png")
        plot(predicted_grid, solid, nx, ny, title="Predicted", save="hat_plot.png")
        plot(error_grid, solid, nx, ny, title="Error", save="err_plot.png")