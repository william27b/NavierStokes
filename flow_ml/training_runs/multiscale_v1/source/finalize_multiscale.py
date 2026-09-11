"""Inspect worst held-out predictions, verify the checkpoint, and record uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from train_multiscale import ROOT,BASELINE,load_new,get_graphs,make_plots,predict,write_json
from model import GNN

def finalize(folder):
    folder=Path(folder)
    report=json.loads((folder/'final_metrics.json').read_text())
    model,_,checkpoint=load_new(folder/'best.pt','cuda')
    model.eval()
    graphs,inventory=get_graphs('test',5,'cuda')
    old=torch.load(BASELINE,map_location='cuda',weights_only=True)
    baseline=GNN(old['hidden'],old['layers'],old['global_context']).cuda().eval()
    baseline.load_state_dict(old['model_state'])
    print('Rendering solver / old / new / error plots for all test cases.',flush=True)
    make_plots(model,graphs,folder/'test_plots',baseline,count=len(graphs))
    rows=report['test']['multiscale']['cases']
    worst_total=max(rows,key=lambda r:r['loss'])
    worst_pressure=max(rows,key=lambda r:r['mse'][2])
    seeds={worst_total['seed'],worst_pressure['seed']}
    make_plots(model,[g for g in graphs if g['seed'] in seeds],folder/'worst_test_plots',baseline,count=len(seeds))
    old_by_seed={r['seed']:r for r in report['test']['old_gnn']['cases']}
    comparisons={str(r['seed']):{'new':r,'old':old_by_seed[r['seed']]} for r in rows if r['seed'] in seeds}
    write_json(folder/'worst_test_cases.json',{'worst_total_seed':worst_total['seed'],
        'worst_pressure_seed':worst_pressure['seed'],'cases':comparisons})
    rng=np.random.default_rng(711)
    samples=rng.integers(0,len(rows),size=(10000,len(rows)))
    intervals={}
    selectors={'total_field_mse':lambda r:r['loss'],
               'pressure_mse':lambda r:r['mse'][2],
               'pressure_gradient_mse':lambda r:r['gradient_mse'][2],
               'far_pressure_mse':lambda r:r['far_pressure_mse']}
    for name,select in selectors.items():
        a=np.asarray([select(old_by_seed[r['seed']]) for r in rows])
        b=np.asarray([select(r) for r in rows])
        ratios=1-b[samples].mean(1)/a[samples].mean(1)
        intervals[name]={'reduction':float(1-b.mean()/a.mean()),
            'paired_bootstrap_95_percent_interval':np.quantile(ratios,[.025,.975]).tolist()}
    write_json(folder/'paired_bootstrap.json',{'resamples':10000,'seed':711,'metrics':intervals})
    benchmark={}
    g=graphs[0]
    with torch.inference_mode():
        for label,m,old_flag in [('old_gnn',baseline,True),('multiscale',model,False)]:
            for _ in range(10):predict(m,g,old_flag)
            torch.cuda.synchronize()
            times=[]
            for _ in range(50):
                start=time.perf_counter()
                predict(m,g,old_flag)
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            benchmark[label]={'median_ms':float(np.median(times)),
                'p90_ms':float(np.quantile(times,.9)),
                'parameters':sum(p.numel() for p in m.parameters())}
    finite_model=all(torch.isfinite(p).all().item() for p in model.parameters())
    states=checkpoint['optimizer_state']['state']
    finite_optimizer=all(torch.isfinite(value).all().item() for state in states.values()
                         for value in state.values() if isinstance(value,torch.Tensor))
    original_hash=hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    assert original_hash==report['baseline_sha256']
    assert inventory['fingerprint']==report['test_inventory']['fingerprint']
    assert finite_model and finite_optimizer
    assert len(states)==len(list(model.parameters()))
    verification={'finite_model':finite_model,'finite_optimizer':finite_optimizer,
        'optimizer_parameter_states':len(states),
        'optimizer_step_counts':sorted({int(state['step']) for state in states.values()}),
        'selected_epoch':checkpoint['completed_epochs'],
        'old_checkpoint_unchanged':True,'test_inventory_unchanged':True,
        'benchmark_device':torch.cuda.get_device_name(),
        'benchmark_note':'Single cached graph, no concurrent training; includes Python model dispatch, excludes graph construction and disk IO.',
        'inference':benchmark,'worst_total_seed':worst_total['seed'],
        'worst_pressure_seed':worst_pressure['seed']}
    write_json(folder/'verification.json',verification)
    print(json.dumps(verification,indent=2))
    print(json.dumps(intervals,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    finalize(parser.parse_args().folder)
