# Solver-run quality audit

Audited 300 saved runs. **300 pass the chosen screen.**

Existing training filter: 300 runs. Nonfinite field archives: 0.

## Chosen acceptance criteria

- Maximum local Mach number <= 0.1, with Ma = sqrt(3) * hypot(ux, uy).
- Maximum absolute density deviation <= 5.0%, measured as max(abs(rho / 1 - 1)).
- Converged, finite fluid fields, positive fluid density, and zero invalid cells in metadata.
- Both physical limits apply to every fluid cell, including inlet/outlet columns; solids are excluded.

Ma <= 0.10 is a conservative low-Mach target. A separate 5% maximum density-excursion budget prevents accepting large density changes even at low speed. These are selected project defaults; the sources motivate low-Mach screening, not a universal numerical cutoff. Explicit function/CLI limits override the defaults.

Passing this screen is not proof of solver accuracy and is not a direct divergence, conservation, resolution-convergence or boundary-condition test. O(Ma^2) scaling does not guarantee a particular percentage error. No thresholds use GNN predictions or loss.

The reference density is 1 by default because the current kernel imposes rho = 1 at the outlet. For another solver configuration, supply the matching reference explicitly.

## Measured distributions

Quantiles below summarize currently training-eligible runs with measurable fields. Each row summarizes one maximum per run; these are not pooled cell statistics.

| Per-run measurement | Minimum | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| All fluid: maximum Mach | 0.027 | 0.037 | 0.063 | 0.093 |
| All fluid: maximum density deviation (%) | 0.333 | 0.375 | 0.583 | 1.977 |
| Without inlet/outlet columns: maximum Mach | 0.027 | 0.037 | 0.063 | 0.093 |
| Without inlet/outlet columns: maximum density deviation (%) | 0.127 | 0.171 | 0.451 | 1.885 |

Prescribed inlet Mach range: 0.017321 to 0.017321.

## Sensitivity to looser limits

These alternatives are diagnostic comparisons, not adopted acceptance criteria.

| Maximum Mach allowed | Maximum density deviation allowed | Qualifying runs |
|---:|---:|---:|
| 0.1 | 5% | 300 |
| 0.2 | 5% | 300 |
| 0.3 | 5% | 300 |
| 0.3 | 10% | 300 |

## Boundary and export stage

300 runs store populations and fields after boundary reconstruction and regularization. 0 runs use the older export of moments before boundary reconstruction.

Measurements excluding x=0 and x=nx-1 remain an additional diagnostic; they do not replace the whole-fluid acceptance check or exclude wall-adjacent cells.

No archives, solver settings, training filters, or checkpoints were changed. The audit does not start training or regenerate cases.

## Files

- `audit.json`: measurements, per-run reasons, acceptance lists, policy and source links.
- `quality_overview.png`: one point per measurable, currently training-eligible run.

## Basis for the screening policy

- [OpenLB: compressibility effects](https://www.openlb.net/forum/reply/5065/): Mach-related incompressible-flow error scales as O(Ma^2).
- [Palabos user guide, chapter 7](https://palabos.unige.ch/files/9515/6509/3036/Palabos_UserGuide.pdf): Standard BGK uses low Mach numbers to limit compressibility effects.
