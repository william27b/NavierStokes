# Training continuation

Continued `/home/william/NavierStokes/flow_ml/model_v4.pt` from epoch 40 through epoch 140 on the same 300 training cases.

Architecture: 64 hidden units, 8 message-passing layers, global context. Adam learning rate: 0.001; batch size: 4; unchanged target normalization and case weighting.

The source checkpoint had no optimizer state. Adam started fresh once, then its state persisted throughout all training blocks.

Every continuation checkpoint includes model weights, Adam state, completed epoch count, and random-generator states. The original checkpoint was preserved.

| Completed epochs | Total normalized MSE | ux MSE | uy MSE | Pressure MSE |
|---|---:|---:|---:|---:|
| 40 | 0.0039889922 | 0.0070051992 | 0.0028458083 | 0.0021159692 |
| 60 | 0.0035341798 | 0.0060560633 | 0.0022476686 | 0.0022988073 |
| 80 | 0.0027137996 | 0.0051432228 | 0.0018968651 | 0.0011013108 |
| 100 | 0.0021255748 | 0.0035605859 | 0.0013328218 | 0.0014833167 |
| 120 | 0.0012071868 | 0.0019560213 | 0.00085442472 | 0.00081111429 |
| 140 | 0.0011663943 | 0.0019548178 | 0.00085466122 | 0.00068970393 |

![Per-channel learning curve](channel_learning_curve.png)

These evaluations use the frozen checkpoint on all training cases; they measure fit, not generalization. Online epoch losses are recorded separately in `history.json`.

Final checkpoint: `/home/william/NavierStokes/flow_ml/training_runs/20260911T082502281765Z_continue_100/checkpoint_epoch_140.pt`

[Final per-channel baseline comparison and five cases with separate target/prediction/error plots](evaluations/epoch_140/summary.md)

[Initial checkpoint evaluation](evaluations/epoch_040/summary.md)

## Pressure halo review

The local message path has eight layers over four-neighbor fluid edges. The pressure plots below use one identical color scale and mark an eight-cell Manhattan-distance contour around the obstacle. The trained model still shows pressure artifacts localized around the obstacle neighborhood. This supports testing greater spatial message reach; it is not a proof of a strict global cutoff, because the model also uses global mean pooling.

- [Solver pressure](pressure_reach_review/seed_1229_solver.png)
- [Epoch 40 pressure](pressure_reach_review/seed_1229_epoch_040.png)
- [Epoch 140 pressure](pressure_reach_review/seed_1229_epoch_140.png)

The continuation reduced total training MSE by 70.76%; 299/300 cases improved relative to epoch 40. The velocity-channel checkpoint errors were nearly unchanged from epochs 120 to 140, while pressure MSE fell further. These aggregate improvements do not establish acceptable spatial accuracy or generalization.

[Checkpoint and optimizer verification](verification.json): all five checkpoints contain the expected Adam update counts (1,500 through 7,500), finite moments, and the fixed learning rate. The original checkpoint hash is unchanged.
