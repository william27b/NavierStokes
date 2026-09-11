# Boundary-aware multiscale GNN experiment

Checkpoint selected using validation only: **epoch 150**. Training: 300 geometries; validation: 40; test: 40.

**This approach substantially improves prediction within the current 64×64 data distribution.** The pressure influence is no longer dominated by the previous fixed-radius halo. Some coarse-grid texture and geometry-specific errors remain.

**Browse every test case:** [interactive gallery](test_gallery.html) · [all 160 image links](test_index.md).

The comparison uses the previous 140-epoch GNN and the same solver targets, normalization, fluid nodes, and equal-case averaging. These results measure the combined architecture, feature, and loss changes; they are not an isolated hierarchy ablation.

## Held-out results

| Error measure | Previous GNN | Multiscale GNN | Reduction |
|---|---:|---:|---:|
| ux field MSE | 0.00211289 | 0.000474631 | 77.5% |
| uy field MSE | 0.001404741 | 0.000296815 | 78.9% |
| pressure field MSE | 0.0004119805 | 0.0001163704 | 71.8% |
| pressure gradient MSE | 0.0001367043 | 6.147575e-05 | 55.0% |
| pressure >8 cells from obstacles MSE | 0.0003660509 | 0.0001006539 | 72.5% |

| Field | Previous lattice RMSE | Multiscale lattice RMSE |
|---|---:|---:|
| ux | 0.002298309 | 0.001089301 |
| uy | 0.001873994 | 0.000861416 |
| pressure | 5.074325e-05 | 2.696877e-05 |

Lower total field MSE on **40/40** test cases; lower pressure MSE on **40/40**.

![Test error ratios](test_error_ratios.png)

![Validation learning curve](validation_learning_curve.png)

## What changed

- Twelve node features: coordinates, inlet/outlet flags, four directional solid-neighbor indicators, distance to solid, prescribed inlet velocity components, and viscosity.
- Five fluid graphs, approximately 64×64 → 32×32 → 16×16 → 8×8 → 4×4. Nodes are merged only through real fluid edges inside each pooling block. Disconnected fluid portions in the same block remain separate. Coarse edges come only from existing finer edges.
- Weighted restriction by represented fluid area, residual connections down and up the hierarchy, and fine-level skip connections. Two local processor steps on each side of each level; eight steps at the coarsest level.
- Exact prescribed inlet velocities and outlet pressure/transverse velocity. No zero-velocity constraint is imposed on fluid nodes next to halfway bounce-back walls.
- Loss: variance-scaled field MSE + 0.15 × target-gradient matching + 0.25 × coarse pressure error. All normalization statistics come from training data only. The gradient term matches solver differences; it does not penalize the prediction gradient itself.
- Cached geometry-only graphs; AdamW, gradient clipping, cosine learning-rate decay, validation selection; checkpoints include optimizer, scheduler, RNG states, loss scales, feature schema, data fingerprints, and source hashes.

## Individual field comparisons

Each geometry has a shared pressure color range and velocity arrow scale for the solver, old GNN, and multiscale GNN. Error figures show signed residuals.

| Seed | Solver | Previous GNN | Multiscale | Error |
|---|---|---|---|---|
| 3000 | [Solver](test_plots/seed_3000_solver.png) | [Old](test_plots/seed_3000_old_gnn.png) | [New](test_plots/seed_3000_multiscale.png) | [Error](test_plots/seed_3000_error.png) |
| 3001 | [Solver](test_plots/seed_3001_solver.png) | [Old](test_plots/seed_3001_old_gnn.png) | [New](test_plots/seed_3001_multiscale.png) | [Error](test_plots/seed_3001_error.png) |
| 3002 | [Solver](test_plots/seed_3002_solver.png) | [Old](test_plots/seed_3002_old_gnn.png) | [New](test_plots/seed_3002_multiscale.png) | [Error](test_plots/seed_3002_error.png) |
| 3003 | [Solver](test_plots/seed_3003_solver.png) | [Old](test_plots/seed_3003_old_gnn.png) | [New](test_plots/seed_3003_multiscale.png) | [Error](test_plots/seed_3003_error.png) |
| 3004 | [Solver](test_plots/seed_3004_solver.png) | [Old](test_plots/seed_3004_old_gnn.png) | [New](test_plots/seed_3004_multiscale.png) | [Error](test_plots/seed_3004_error.png) |
| 3005 | [Solver](test_plots/seed_3005_solver.png) | [Old](test_plots/seed_3005_old_gnn.png) | [New](test_plots/seed_3005_multiscale.png) | [Error](test_plots/seed_3005_error.png) |
| 3006 | [Solver](test_plots/seed_3006_solver.png) | [Old](test_plots/seed_3006_old_gnn.png) | [New](test_plots/seed_3006_multiscale.png) | [Error](test_plots/seed_3006_error.png) |
| 3007 | [Solver](test_plots/seed_3007_solver.png) | [Old](test_plots/seed_3007_old_gnn.png) | [New](test_plots/seed_3007_multiscale.png) | [Error](test_plots/seed_3007_error.png) |
| 3008 | [Solver](test_plots/seed_3008_solver.png) | [Old](test_plots/seed_3008_old_gnn.png) | [New](test_plots/seed_3008_multiscale.png) | [Error](test_plots/seed_3008_error.png) |
| 3009 | [Solver](test_plots/seed_3009_solver.png) | [Old](test_plots/seed_3009_old_gnn.png) | [New](test_plots/seed_3009_multiscale.png) | [Error](test_plots/seed_3009_error.png) |
| 3010 | [Solver](test_plots/seed_3010_solver.png) | [Old](test_plots/seed_3010_old_gnn.png) | [New](test_plots/seed_3010_multiscale.png) | [Error](test_plots/seed_3010_error.png) |
| 3011 | [Solver](test_plots/seed_3011_solver.png) | [Old](test_plots/seed_3011_old_gnn.png) | [New](test_plots/seed_3011_multiscale.png) | [Error](test_plots/seed_3011_error.png) |
| 3012 | [Solver](test_plots/seed_3012_solver.png) | [Old](test_plots/seed_3012_old_gnn.png) | [New](test_plots/seed_3012_multiscale.png) | [Error](test_plots/seed_3012_error.png) |
| 3013 | [Solver](test_plots/seed_3013_solver.png) | [Old](test_plots/seed_3013_old_gnn.png) | [New](test_plots/seed_3013_multiscale.png) | [Error](test_plots/seed_3013_error.png) |
| 3014 | [Solver](test_plots/seed_3014_solver.png) | [Old](test_plots/seed_3014_old_gnn.png) | [New](test_plots/seed_3014_multiscale.png) | [Error](test_plots/seed_3014_error.png) |
| 3015 | [Solver](test_plots/seed_3015_solver.png) | [Old](test_plots/seed_3015_old_gnn.png) | [New](test_plots/seed_3015_multiscale.png) | [Error](test_plots/seed_3015_error.png) |
| 3016 | [Solver](test_plots/seed_3016_solver.png) | [Old](test_plots/seed_3016_old_gnn.png) | [New](test_plots/seed_3016_multiscale.png) | [Error](test_plots/seed_3016_error.png) |
| 3017 | [Solver](test_plots/seed_3017_solver.png) | [Old](test_plots/seed_3017_old_gnn.png) | [New](test_plots/seed_3017_multiscale.png) | [Error](test_plots/seed_3017_error.png) |
| 3018 | [Solver](test_plots/seed_3018_solver.png) | [Old](test_plots/seed_3018_old_gnn.png) | [New](test_plots/seed_3018_multiscale.png) | [Error](test_plots/seed_3018_error.png) |
| 3019 | [Solver](test_plots/seed_3019_solver.png) | [Old](test_plots/seed_3019_old_gnn.png) | [New](test_plots/seed_3019_multiscale.png) | [Error](test_plots/seed_3019_error.png) |
| 3020 | [Solver](test_plots/seed_3020_solver.png) | [Old](test_plots/seed_3020_old_gnn.png) | [New](test_plots/seed_3020_multiscale.png) | [Error](test_plots/seed_3020_error.png) |
| 3021 | [Solver](test_plots/seed_3021_solver.png) | [Old](test_plots/seed_3021_old_gnn.png) | [New](test_plots/seed_3021_multiscale.png) | [Error](test_plots/seed_3021_error.png) |
| 3022 | [Solver](test_plots/seed_3022_solver.png) | [Old](test_plots/seed_3022_old_gnn.png) | [New](test_plots/seed_3022_multiscale.png) | [Error](test_plots/seed_3022_error.png) |
| 3023 | [Solver](test_plots/seed_3023_solver.png) | [Old](test_plots/seed_3023_old_gnn.png) | [New](test_plots/seed_3023_multiscale.png) | [Error](test_plots/seed_3023_error.png) |
| 3024 | [Solver](test_plots/seed_3024_solver.png) | [Old](test_plots/seed_3024_old_gnn.png) | [New](test_plots/seed_3024_multiscale.png) | [Error](test_plots/seed_3024_error.png) |
| 3025 | [Solver](test_plots/seed_3025_solver.png) | [Old](test_plots/seed_3025_old_gnn.png) | [New](test_plots/seed_3025_multiscale.png) | [Error](test_plots/seed_3025_error.png) |
| 3026 | [Solver](test_plots/seed_3026_solver.png) | [Old](test_plots/seed_3026_old_gnn.png) | [New](test_plots/seed_3026_multiscale.png) | [Error](test_plots/seed_3026_error.png) |
| 3027 | [Solver](test_plots/seed_3027_solver.png) | [Old](test_plots/seed_3027_old_gnn.png) | [New](test_plots/seed_3027_multiscale.png) | [Error](test_plots/seed_3027_error.png) |
| 3028 | [Solver](test_plots/seed_3028_solver.png) | [Old](test_plots/seed_3028_old_gnn.png) | [New](test_plots/seed_3028_multiscale.png) | [Error](test_plots/seed_3028_error.png) |
| 3029 | [Solver](test_plots/seed_3029_solver.png) | [Old](test_plots/seed_3029_old_gnn.png) | [New](test_plots/seed_3029_multiscale.png) | [Error](test_plots/seed_3029_error.png) |
| 3030 | [Solver](test_plots/seed_3030_solver.png) | [Old](test_plots/seed_3030_old_gnn.png) | [New](test_plots/seed_3030_multiscale.png) | [Error](test_plots/seed_3030_error.png) |
| 3031 | [Solver](test_plots/seed_3031_solver.png) | [Old](test_plots/seed_3031_old_gnn.png) | [New](test_plots/seed_3031_multiscale.png) | [Error](test_plots/seed_3031_error.png) |
| 3032 | [Solver](test_plots/seed_3032_solver.png) | [Old](test_plots/seed_3032_old_gnn.png) | [New](test_plots/seed_3032_multiscale.png) | [Error](test_plots/seed_3032_error.png) |
| 3033 | [Solver](test_plots/seed_3033_solver.png) | [Old](test_plots/seed_3033_old_gnn.png) | [New](test_plots/seed_3033_multiscale.png) | [Error](test_plots/seed_3033_error.png) |
| 3034 | [Solver](test_plots/seed_3034_solver.png) | [Old](test_plots/seed_3034_old_gnn.png) | [New](test_plots/seed_3034_multiscale.png) | [Error](test_plots/seed_3034_error.png) |
| 3035 | [Solver](test_plots/seed_3035_solver.png) | [Old](test_plots/seed_3035_old_gnn.png) | [New](test_plots/seed_3035_multiscale.png) | [Error](test_plots/seed_3035_error.png) |
| 3036 | [Solver](test_plots/seed_3036_solver.png) | [Old](test_plots/seed_3036_old_gnn.png) | [New](test_plots/seed_3036_multiscale.png) | [Error](test_plots/seed_3036_error.png) |
| 3037 | [Solver](test_plots/seed_3037_solver.png) | [Old](test_plots/seed_3037_old_gnn.png) | [New](test_plots/seed_3037_multiscale.png) | [Error](test_plots/seed_3037_error.png) |
| 3038 | [Solver](test_plots/seed_3038_solver.png) | [Old](test_plots/seed_3038_old_gnn.png) | [New](test_plots/seed_3038_multiscale.png) | [Error](test_plots/seed_3038_error.png) |
| 3039 | [Solver](test_plots/seed_3039_solver.png) | [Old](test_plots/seed_3039_old_gnn.png) | [New](test_plots/seed_3039_multiscale.png) | [Error](test_plots/seed_3039_error.png) |

The original reported halo geometry is revisited in [solver](original_halo_case/seed_1229_solver.png), [old GNN](original_halo_case/seed_1229_old_gnn.png), and [multiscale GNN](original_halo_case/seed_1229_multiscale.png).

A supplementary [pressure-only view](pressure_detail/train_seed_1229/multiscale_gnn.png) and [channel pressure-drop profile](pressure_detail/train_seed_1229/channel_pressure_profile.png) show the domain-scale pressure structure. The pressure-only map uses explicitly marked percentile color limits to resolve interior variation.

**Worst test case: seed 3008.** Compare its [solver](test_plots/seed_3008_solver.png), [new prediction](test_plots/seed_3008_multiscale.png), and [signed error](test_plots/seed_3008_error.png). It still has appreciable velocity-direction and pressure errors around the two obstacles despite improving over the old model. This result does not establish solver-level accuracy for every geometry.

## Verification and runtime

Nine tests passed: wall/detour preservation, disconnected-region separation, conservative pooling, boundary conditions, permutation equivariance, loss behavior, a distant gradient path, exact CPU optimizer/scheduler restart, and inference independence from target labels. All 380 geometry masks are distinct. Final weights and all 280 optimizer parameter states are finite; the selected checkpoint contains 11,250 optimizer updates. The old checkpoint is unchanged.

Paired bootstrap intervals over the 40 test cases put the field-MSE reduction at 73–83% and pressure-MSE reduction at 62–80% (95% intervals; 10,000 resamples). Details are in `paired_bootstrap.json`.

On the RTX 4090, median inference on one cached test graph was 8.1 ms for the multiscale model versus 2.5 ms for the old model; graph construction and file IO are excluded. Parameter counts are 697,731 versus 241,155. This experiment prioritizes spatial accuracy; the new architecture costs more per prediction.

## Scope

The held-out cases use the same 64×64 resolution, geometry sampler, ux=0.01 and tau=0.53. This tests generalization to new geometries within that distribution. It does not establish accuracy at other Reynolds numbers, resolutions, or fundamentally different geometries. Coarsening preserves fluid connectivity but approximates sub-block travel distances; fine skip connections retain local detail.

Raw per-case results and dataset hashes: `final_metrics.json`. Training history: `history.json`. Best model: `best.pt`. Last resumable state: `last.pt`.
