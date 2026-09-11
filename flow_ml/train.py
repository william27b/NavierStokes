from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import load_case, build_edges, to_torch, FlowDataset
from model import GNN

from utils import plot
import json
import os

def train(model, loader, device, epochs=1_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        total_loss = 0.0

        for inputs in loader:
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }
            optimizer.zero_grad(set_to_none=True)

            prediction = model(
                inputs["x"],
                inputs["edge_index"],
                inputs["edge_attr"],
            )

            loss = (prediction - inputs["y"]).square().mean()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: {total_loss / len(loader):.6f}")

if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent.parent / "flow_ml/solver_runs"

    cases = []

    for metadata_path in sorted(dataset_dir.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())

        if (
            metadata["converged"]
            and metadata["invalid_fluid_cells"] == 0
        ):
            cases.append(metadata_path.parent)

    print(f"Training on {len(cases)} cases")

    dataset = FlowDataset(cases)
    loader = DataLoader(dataset, batch_size=None, shuffle=True)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(hidden=64, layers=4).to(device)
    model.train()

    train(model, loader, device, epochs=10)

    torch.save(
        {
            "model_state": model.state_dict(),
            "hidden": 64,
            "layers": 4,
        },
        Path(__file__).with_name("model_v2.pt"),
    )