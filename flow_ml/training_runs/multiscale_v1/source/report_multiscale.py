"""Make a compact, reproducible comparison after the final held-out evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(root):
    root = Path(root)
    data = json.loads((root/'final_metrics.json').read_text())
    history = json.loads((root/'history.json').read_text())
    chosen = data['selected_epoch']
    rows = [r for r in history if 'validation' in r]
    epochs = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(figsize=(10,5.5),layout='constrained')
    colors = ['#2563eb','#d97706','#059669']
    for i,(name,color) in enumerate(zip(['ux','uy','pressure'],colors)):
        ax.plot(epochs,[r['validation']['mse'][i] for r in rows],label=f'Multiscale {name}',color=color)
        ax.axhline(data['validation']['old_gnn']['summary']['mse'][i],color=color,linestyle='--',alpha=.55)
    ax.axvline(chosen,color='#444',linestyle=':',label=f'Selected epoch {chosen}')
    ax.set(yscale='log',xlabel='Training epoch',ylabel='Case-averaged normalized field MSE',
           title='Unseen validation geometries | dashed lines: old GNN')
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(root/'validation_learning_curve.png',dpi=160);plt.close(fig)
    old = data['test']['old_gnn']['summary']
    new = data['test']['multiscale']['summary']
    fig,ax = plt.subplots(figsize=(10,5.5),layout='constrained')
    names = ['ux field','uy field','pressure field','pressure gradient','pressure >8 cells\nfrom obstacles']
    old_values = old['mse'] + [old['gradient_mse'][2],old['far_pressure_mse']]
    new_values = new['mse'] + [new['gradient_mse'][2],new['far_pressure_mse']]
    x=np.arange(5)
    ratios=np.asarray(new_values)/old_values
    ax.bar(x,ratios,color=['#2563eb','#2563eb','#059669','#059669','#059669'])
    ax.axhline(1,color='#64748b',linestyle='--',label='Old GNN error')
    for i,ratio in enumerate(ratios):
        ax.text(i,ratio+.02,f'{100*ratio:.1f}%',ha='center')
    ax.set_xticks(x,names)
    ax.set(ylabel='New error / old error (lower is better)',ylim=(0,max(1.15,float(max(ratios))*1.15)),
           title=f'Final test: {new["case_count"]} unseen geometries')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'test_error_ratios.png',dpi=160);plt.close(fig)
    lines=['# Boundary-aware multiscale GNN experiment','',
        f'Checkpoint selected using validation only: **epoch {chosen}**. Training: 300 geometries; validation: 40; test: 40.',
        '', '**This approach substantially improves prediction within the current 64×64 data distribution.** The pressure influence is no longer dominated by the previous fixed-radius halo. Some coarse-grid texture and geometry-specific errors remain.',
        '', '**Browse every test case:** [interactive gallery](test_gallery.html) · [all 160 image links](test_index.md).',
        '', 'The comparison uses the previous 140-epoch GNN and the same solver targets, normalization, fluid nodes, and equal-case averaging. These results measure the combined architecture, feature, and loss changes; they are not an isolated hierarchy ablation.',
        '', '## Held-out results','',
        '| Error measure | Previous GNN | Multiscale GNN | Reduction |','|---|---:|---:|---:|']
    for name, a, b in zip(names,old_values,new_values):
        lines.append(f'| {name.replace(chr(10)," ")} MSE | {a:.7g} | {b:.7g} | {100*(1-b/a):.1f}% |')
    lines += ['', '| Field | Previous lattice RMSE | Multiscale lattice RMSE |', '|---|---:|---:|']
    for name,a,b in zip(['ux','uy','pressure'],old['lattice_rmse'],new['lattice_rmse']):
        lines.append(f'| {name} | {a:.7g} | {b:.7g} |')
    old_cases={r['seed']:r for r in data['test']['old_gnn']['cases']}
    new_cases=data['test']['multiscale']['cases']
    wins=sum(r['loss']<old_cases[r['seed']]['loss'] for r in new_cases)
    p_wins=sum(r['mse'][2]<old_cases[r['seed']]['mse'][2] for r in new_cases)
    lines += ['',f'Lower total field MSE on **{wins}/{len(new_cases)}** test cases; lower pressure MSE on **{p_wins}/{len(new_cases)}**.',
        '', '![Test error ratios](test_error_ratios.png)','', '![Validation learning curve](validation_learning_curve.png)',
        '', '## What changed','',
        '- Twelve node features: coordinates, inlet/outlet flags, four directional solid-neighbor indicators, distance to solid, prescribed inlet velocity components, and viscosity.',
        '- Five fluid graphs, approximately 64×64 → 32×32 → 16×16 → 8×8 → 4×4. Nodes are merged only through real fluid edges inside each pooling block. Disconnected fluid portions in the same block remain separate. Coarse edges come only from existing finer edges.',
        '- Weighted restriction by represented fluid area, residual connections down and up the hierarchy, and fine-level skip connections. Two local processor steps on each side of each level; eight steps at the coarsest level.',
        '- Exact prescribed inlet velocities and outlet pressure/transverse velocity. No zero-velocity constraint is imposed on fluid nodes next to halfway bounce-back walls.',
        '- Loss: variance-scaled field MSE + 0.15 × target-gradient matching + 0.25 × coarse pressure error. All normalization statistics come from training data only. The gradient term matches solver differences; it does not penalize the prediction gradient itself.',
        '- Cached geometry-only graphs; AdamW, gradient clipping, cosine learning-rate decay, validation selection; checkpoints include optimizer, scheduler, RNG states, loss scales, feature schema, data fingerprints, and source hashes.',
        '', '## Individual field comparisons','',
        'Each geometry has a shared pressure color range and velocity arrow scale for the solver, old GNN, and multiscale GNN. Error figures show signed residuals.',
        '', '| Seed | Solver | Previous GNN | Multiscale | Error |','|---|---|---|---|---|']
    seeds=sorted(int(p.stem.split('_')[1]) for p in (root/'test_plots').glob('*_solver.png'))
    for seed in seeds:
        lines.append(f'| {seed} | [Solver](test_plots/seed_{seed}_solver.png) | [Old](test_plots/seed_{seed}_old_gnn.png) | [New](test_plots/seed_{seed}_multiscale.png) | [Error](test_plots/seed_{seed}_error.png) |')
    lines += ['', 'The original reported halo geometry is revisited in [solver](original_halo_case/seed_1229_solver.png), [old GNN](original_halo_case/seed_1229_old_gnn.png), and [multiscale GNN](original_halo_case/seed_1229_multiscale.png).',
        '', 'A supplementary [pressure-only view](pressure_detail/train_seed_1229/multiscale_gnn.png) and [channel pressure-drop profile](pressure_detail/train_seed_1229/channel_pressure_profile.png) show the domain-scale pressure structure. The pressure-only map uses explicitly marked percentile color limits to resolve interior variation.',
        '', '**Worst test case: seed 3008.** Compare its [solver](test_plots/seed_3008_solver.png), [new prediction](test_plots/seed_3008_multiscale.png), and [signed error](test_plots/seed_3008_error.png). It still has appreciable velocity-direction and pressure errors around the two obstacles despite improving over the old model. This result does not establish solver-level accuracy for every geometry.',
        '', '## Verification and runtime','',
        'Nine tests passed: wall/detour preservation, disconnected-region separation, conservative pooling, boundary conditions, permutation equivariance, loss behavior, a distant gradient path, exact CPU optimizer/scheduler restart, and inference independence from target labels. All 380 geometry masks are distinct. Final weights and all 280 optimizer parameter states are finite; the selected checkpoint contains 11,250 optimizer updates. The old checkpoint is unchanged.',
        '', 'Paired bootstrap intervals over the 40 test cases put the field-MSE reduction at 73–83% and pressure-MSE reduction at 62–80% (95% intervals; 10,000 resamples). Details are in `paired_bootstrap.json`.',
        '', 'On the RTX 4090, median inference on one cached test graph was 8.1 ms for the multiscale model versus 2.5 ms for the old model; graph construction and file IO are excluded. Parameter counts are 697,731 versus 241,155. This experiment prioritizes spatial accuracy; the new architecture costs more per prediction.',
        '', '## Scope','',
        'The held-out cases use the same 64×64 resolution, geometry sampler, ux=0.01 and tau=0.53. This tests generalization to new geometries within that distribution. It does not establish accuracy at other Reynolds numbers, resolutions, or fundamentally different geometries. Coarsening preserves fluid connectivity but approximates sub-block travel distances; fine skip connections retain local detail.',
        '', 'Raw per-case results and dataset hashes: `final_metrics.json`. Training history: `history.json`. Best model: `best.pt`. Last resumable state: `last.pt`.','']
    (root/'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'test_total_reduction':1-new['loss']/old['loss'],
                      'test_pressure_reduction':1-new['mse'][2]/old['mse'][2],
                      'test_gradient_pressure_reduction':1-new['gradient_mse'][2]/old['gradient_mse'][2],
                      'case_wins':wins,'pressure_wins':p_wins},indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
