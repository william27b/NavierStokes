"""Continue a flow checkpoint, retaining Adam state and evaluating every block."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import FlowDataset
from evaluate_dataset import eligible_runs, evaluate_dataset
from model import GNN
from train import train


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_inventory(runs):
    records = []
    for directory, metadata in runs:
        files = {}
        for name in ("metadata.json", "fields.npz"):
            path = directory/name
            stat = path.stat()
            files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        records.append({"run_dir": str(directory), "seed": metadata.get("generation", {}).get("seed"), "files": files})
    encoded = json.dumps(records, sort_keys=True).encode()
    return records, hashlib.sha256(encoded).hexdigest()


def restore_training(checkpoint, device, *, seed=0):
    memory = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = {"hidden": memory["hidden"], "layers": memory["layers"],
              "global_context": memory.get("global_context", False)}
    torch.manual_seed(seed)
    model = GNN(**config).to(device)
    model.load_state_dict(memory["model_state"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    restored_optimizer = "optimizer_state" in memory
    if restored_optimizer:
        optimizer.load_state_dict(memory["optimizer_state"])
    generator = torch.Generator().manual_seed(seed)
    if "loader_rng_state" in memory:
        generator.set_state(memory["loader_rng_state"])
    if "torch_rng_state" in memory:
        torch.set_rng_state(memory["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_states" in memory:
        torch.cuda.set_rng_state_all(memory["cuda_rng_states"])
    return model, optimizer, generator, memory, restored_optimizer


def save_checkpoint(path, model, optimizer, generator, completed_epochs, *, details=None):
    payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
               "hidden": model.encoder[0].out_features, "layers": len(model.processors),
               "global_context": model.global_context, "completed_epochs": completed_epochs,
               "loader_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
               "details": details or {}}
    if next(model.parameters()).is_cuda:
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(history, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=2, allow_nan=False))
    temporary.replace(path)


def write_summary(history, output):
    checkpoints = history["evaluations"]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    epoch_numbers = [item["epoch"] for item in checkpoints]
    for channel, label, color in zip(range(3), ("ux", "uy", "pressure"), ("#2563eb", "#d97706", "#059669")):
        values = [item["metrics"]["model"]["normalized_mse"][channel] for item in checkpoints]
        ax.plot(epoch_numbers, values, "o-", label=label, color=color)
    ax.set(xlabel="Completed epochs", ylabel="Frozen-checkpoint normalized MSE",
           title=f"Continued training on {history['case_count']} cases | fixed architecture and learning rate")
    ax.set_xticks(epoch_numbers)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(output/"channel_learning_curve.png", dpi=160)
    plt.close(fig)

    first, last = checkpoints[0], checkpoints[-1]
    lines = ["# Training continuation", "",
             f"Continued `{history['source_checkpoint']}` from epoch {history['starting_epoch']} through epoch {last['epoch']} on the same {history['case_count']} training cases.", "",
             "Architecture: 64 hidden units, 8 message-passing layers, global context. Adam learning rate: 0.001; batch size: 4; unchanged target normalization and case weighting.", "",
             ("The source checkpoint had no optimizer state. Adam started fresh once, then its state persisted throughout all training blocks."
              if not history["source_optimizer_restored"] else "Adam state was restored from the source checkpoint and persisted throughout all training blocks."), "",
             "Every continuation checkpoint includes model weights, Adam state, completed epoch count, and random-generator states. The original checkpoint was preserved.", "",
             "| Completed epochs | Total normalized MSE | ux MSE | uy MSE | Pressure MSE |", "|---|---:|---:|---:|---:|"]
    for item in checkpoints:
        model = item["metrics"]["model"]
        lines.append(f"| {item['epoch']} | {model['loss']:.8g} | " + " | ".join(f"{v:.8g}" for v in model["normalized_mse"]) + " |")
    lines += ["", "![Per-channel learning curve](channel_learning_curve.png)", "",
              "These evaluations use the frozen checkpoint on all training cases; they measure fit, not generalization. Online epoch losses are recorded separately in `history.json`.", "",
              f"Final checkpoint: `{last['checkpoint']}`", "",
              f"[Final per-channel baseline comparison and five cases with separate target/prediction/error plots](evaluations/epoch_{last['epoch']:03d}/summary.md)", "",
              f"[Initial checkpoint evaluation](evaluations/epoch_{first['epoch']:03d}/summary.md)", ""]
    (output/"summary.md").write_text("\n".join(lines))


def continue_training(checkpoint=None, *, epochs=100, start_epoch=None, eval_every=20,
                      run_root=None, output_dir=None, seed=0, expected_cases=300, device=None):
    if epochs < 1 or eval_every < 1:
        raise ValueError("epochs and eval_every must be positive")
    base = Path(__file__).resolve().parent
    checkpoint = (base/"model_v4.pt" if checkpoint is None else Path(checkpoint)).resolve()
    run_root = (base/"solver_runs" if run_root is None else Path(run_root)).resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    source_hash = file_hash(checkpoint)
    model, optimizer, generator, memory, restored = restore_training(checkpoint, device, seed=seed)
    stored_epoch = memory.get("completed_epochs")
    if stored_epoch is not None and start_epoch is not None and stored_epoch != start_epoch:
        raise ValueError("Requested start epoch disagrees with checkpoint")
    if start_epoch is None:
        if stored_epoch is None:
            raise ValueError("Legacy checkpoint requires an explicit --start-epoch")
        start_epoch = stored_epoch
    if start_epoch < 0:
        raise ValueError("start_epoch must be nonnegative")
    # This experiment specifically preserves the existing architecture and LR.
    if (memory['hidden'], memory['layers'], memory.get('global_context', False)) != (64, 8, True):
        raise ValueError("This continuation expects the current 64-wide, 8-layer global-context model")
    if any(group['lr'] != 1e-3 for group in optimizer.param_groups):
        raise ValueError("This experiment expects Adam learning rate 0.001")
    runs = eligible_runs(run_root)
    if len(runs) != expected_cases:
        raise ValueError(f"Expected {expected_cases} cases, found {len(runs)}")
    inventory, inventory_hash = dataset_inventory(runs)
    previous_inventory = memory.get("details", {}).get("dataset_inventory_sha256")
    if previous_inventory is not None and previous_inventory != inventory_hash:
        raise ValueError("Dataset differs from the resumed checkpoint's inventory")
    dataset = FlowDataset([path for path, _ in runs])
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=list, generator=generator)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (base/"training_runs"/f"{stamp}_continue" if output_dir is None else Path(output_dir)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(checkpoint, output/"source_checkpoint.pt")
    history = {"source_checkpoint": str(checkpoint), "source_checkpoint_sha256": source_hash,
               "source_optimizer_restored": restored, "starting_epoch": start_epoch,
               "requested_additional_epochs": epochs, "case_count": len(runs), "device": str(device),
               "dataset_inventory_sha256": inventory_hash, "dataset_inventory": inventory,
               "model_config": {"hidden": 64, "layers": 8, "global_context": True},
               "learning_rate": 1e-3, "batch_size": 4, "online_epochs": [], "evaluations": []}
    print(f"Output: {output}", flush=True)
    print(f"Continuing epochs {start_epoch+1} to {start_epoch+epochs} on {len(runs)} cases. Adam restored: {restored}", flush=True)

    def record_epoch(number, loss):
        history["online_epochs"].append({"epoch": number, "loss": loss})
        write_history(history, output/"history.json")

    def evaluate(path, completed):
        # Evaluator construction consumes random numbers; keep training's RNG
        # state intact so evaluation frequency does not change its trajectory.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            report = evaluate_dataset(path, run_root, output_dir=output/"evaluations"/f"epoch_{completed:03d}",
                                      count=5, seed=42, device=str(device))
        history["evaluations"].append({"epoch": completed, "checkpoint": str(path),
                                       "checkpoint_sha256": file_hash(path), "metrics": report["metrics"],
                                       "case_wins": report["case_wins"]})
        write_history(history, output/"history.json")
        write_summary(history, output)
        print(f"CHECKPOINT EVALUATION epoch={completed}: {report['metrics']['model']}", flush=True)

    evaluate(output/"source_checkpoint.pt", start_epoch)
    completed = start_epoch
    finish = start_epoch + epochs
    while completed < finish:
        block = min(eval_every, finish-completed)
        returned = train(model, loader, device, epochs=block, optimizer=optimizer,
                         start_epoch=completed, on_epoch_end=record_epoch)
        if returned is not optimizer:
            raise AssertionError("Training replaced the restored optimizer")
        completed += block
        _, current_inventory_hash = dataset_inventory(eligible_runs(run_root))
        if inventory_hash != current_inventory_hash:
            raise RuntimeError("Dataset changed during training")
        saved = output/f"checkpoint_epoch_{completed:03d}.pt"
        save_checkpoint(saved, model, optimizer, generator, completed,
                        details={"dataset_inventory_sha256": inventory_hash,
                                 "source_checkpoint_sha256": source_hash})
        evaluate(saved, completed)
    if file_hash(checkpoint) != source_hash:
        raise RuntimeError("Source checkpoint changed during this run")
    history["completed_epochs"] = completed
    history["source_checkpoint_unchanged"] = True
    history["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_history(history, output/"history.json")
    write_summary(history, output)
    print(f"COMPLETE: {output/'summary.md'}", flush=True)
    return history, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-cases", type=int, default=300)
    parser.add_argument("--device")
    args = parser.parse_args()
    continue_training(args.checkpoint, epochs=args.epochs, start_epoch=args.start_epoch,
                      eval_every=args.eval_every, run_root=args.run_root, output_dir=args.output_dir,
                      seed=args.seed, expected_cases=args.expected_cases, device=args.device)
