# Frozen-checkpoint training evaluation

Checkpoint: `/home/william/NavierStokes/flow_ml/model_v4.pt`

Evaluated **300 cases** without parameter updates. These are training-fit measurements, not validation results.

Each case gets equal weight: average squared error over its fluid nodes, then average cases. Solids are excluded; fluid inlet/outlet cells are included. Channel RMSE is the square root of this averaged MSE, not the average of case RMSEs.

Normalization: ux/0.05, uy/0.05, (p - 1/3)/0.0025, unchanged from training.

| Method | Total normalized MSE | ux MSE | uy MSE | Pressure MSE |
|---|---:|---:|---:|---:|
| model | 0.0039889922 | 0.0070051992 | 0.0028458083 | 0.0021159692 |
| inlet_constant | 0.0080310068 | 0.013989981 | 0.0030346291 | 0.0070684099 |
| label_fitted_constant | 0.007369149 | 0.013836965 | 0.0030341847 | 0.0052362977 |

| Method | ux RMSE (lattice) | uy RMSE (lattice) | Pressure RMSE (lattice) |
|---|---:|---:|---:|
| model | 0.0041848534 | 0.0026673059 | 0.00011499916 |
| inlet_constant | 0.0059139626 | 0.0027543734 | 0.00021018459 |
| label_fitted_constant | 0.0058815314 | 0.0027541717 | 0.00018090567 |

The inlet constant uses each archive's prescribed inlet velocity, uniformly over fluid nodes, and p=1/3. It uses no target labels. The label-fitted constant is one three-component vector fitted across this evaluated dataset using equal case weights; it is an in-sample diagnostic reference, not a held-out result.

Label-fitted constant [ux, uy, p] in lattice units: `[0.010618499380769007, -3.333222477012595e-05, 0.3334403412829817]`.

Model beats the inlet constant on 300/300 cases and the label-fitted constant on 300/300 cases (total normalized MSE).

The reported epoch loss averages predictions made while parameters change. This report uses only the final checkpoint, so its loss need not equal the epoch-40 log.

![Channel comparison](metrics_comparison.png)

## Reproducibly random cases

Target and prediction plots share pressure and velocity scales across the selection. Signed-error plots use their own shared scales, with pressure centered at zero. Error arrows represent prediction minus target, not a flow solution.

| Seed | Solver target | GNN prediction | Signed error |
|---|---|---|---|
| 1229 | [Target](case_229_seed_1229_target.png) | [Prediction](case_229_seed_1229_prediction.png) | [Error](case_229_seed_1229_error.png) |
| 1131 | [Target](case_131_seed_1131_target.png) | [Prediction](case_131_seed_1131_prediction.png) | [Error](case_131_seed_1131_error.png) |
| 1195 | [Target](case_195_seed_1195_target.png) | [Prediction](case_195_seed_1195_prediction.png) | [Error](case_195_seed_1195_error.png) |
| 1026 | [Target](case_026_seed_1026_target.png) | [Prediction](case_026_seed_1026_prediction.png) | [Error](case_026_seed_1026_error.png) |
| 1129 | [Target](case_129_seed_1129_target.png) | [Prediction](case_129_seed_1129_prediction.png) | [Error](case_129_seed_1129_error.png) |

## Verification

- Every graph target was checked against its raw archived velocity and pressure after normalization.
- Selected graph fields were mapped back using explicit coordinates; solids remain masked.
- Model state tensors and checkpoint SHA-256 were identical before and after evaluation.
- Per-case metrics, checkpoint identity, architecture, selection, and scaling details are in `metrics.json`.
