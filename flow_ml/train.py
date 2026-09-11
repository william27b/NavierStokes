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

def train(model, loader, device, epochs=40, *, optimizer=None, start_epoch=0,
          on_epoch_end=None):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        case_count = 0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            for inputs in batch:
                inputs = {
                    key: value.to(device)
                    for key, value in inputs.items()
                }

                prediction = model(
                    inputs["x"],
                    inputs["edge_index"],
                    inputs["edge_attr"],
                )

                loss = (prediction - inputs["y"]).square().mean()
                (loss / len(batch)).backward()

                total_loss += loss.item()
                case_count += 1


            optimizer.step()

        epoch_number = start_epoch + epoch + 1
        mean_loss = total_loss / case_count
        print(f"Epoch {epoch_number}: {mean_loss:.6f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(epoch_number, mean_loss)

    return optimizer

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
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=list,)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden, layers = 64, 8
    model = GNN(hidden=hidden, layers=layers, global_context=True).to(device)
    model.train()

    optimizer = train(model, loader, device, epochs=40)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "hidden": hidden,
            "layers": layers,
            "global_context": model.global_context,
        },
        Path(__file__).with_name("model_v4.pt"),
    )
