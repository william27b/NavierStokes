"""Train the boundary-aware hierarchy; select on validation, reserve test for final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from model import GNN
from multiscale_graph import load_graph, to_device, pool, SCALE, OFFSET, FEATURES
from multiscale_model import MultiscaleGNN, FlowLoss, fit_loss_scales

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT/'training_runs/20260911T082502281765Z_continue_100/checkpoint_epoch_140.pt'

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)

def get_graphs(split, levels, device):
    root = ROOT/'solver_runs' if split == 'train' else ROOT/'heldout_runs'/split
    runs, records = [], []
    for path in sorted(root.glob('*/metadata.json')):
        meta = json.loads(path.read_text())
        if meta['converged'] and meta['invalid_fluid_cells'] == 0:
            f = path.parent/'fields.npz'
            runs.append(path.parent)
            records.append({'path': str(path.parent), 'seed': meta['generation']['seed'],
                            'fields_sha256': hashlib.sha256(f.read_bytes()).hexdigest()})
    if not runs:
        raise ValueError(f'No eligible {split} runs')
    fingerprint = hashlib.sha256(json.dumps(records,sort_keys=True).encode() +
        (ROOT/'multiscale_graph.py').read_bytes()).hexdigest()
    cache_dir = ROOT/'graph_cache'
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir/f'{split}_{levels}_{fingerprint}.pt'
    if cache.exists():
        graphs = torch.load(cache, weights_only=True)
    else:
        graphs = [load_graph(path, levels) for path in runs]
        torch.save(graphs, cache)
    return [to_device(g, device) for g in graphs], {'fingerprint': fingerprint, 'runs': records}

def predict(model, graph, baseline=False):
    if baseline:
        level = graph['levels'][0]
        return model(graph['x'][:,:2], level['edges'], level['edge_attr'])
    return model(graph)

@torch.inference_mode()
def evaluate(model, graphs, loss_fn, baseline=False):
    model.eval()
    rows = []
    for g in graphs:
        pred = predict(model, g, baseline)
        if not torch.isfinite(pred).all():
            raise ValueError('Nonfinite evaluation prediction')
        error = (pred-g['y']).double()
        a,b = g['levels'][0]['edges']
        mse = error.square().mean(0)
        grad = (error[b]-error[a]).square().mean(0)
        value, parts = loss_fn(pred, g)
        coarse = error[:,2:3]
        for fine, parent in zip(g['levels'][:-1],g['levels'][1:]):
            coarse = pool(coarse, fine, parent)
        weights = g['levels'][-1]['weights']
        coarse_mse = (coarse[:,0].square()*weights).sum()/weights.sum()
        row = {'seed': g['seed'], 'mse': mse.cpu().tolist(), 'gradient_mse': grad.cpu().tolist(),
               'loss': float(mse.mean()), 'objective': float(value), 'parts': parts.cpu().tolist(),
               'coarse_pressure_mse': float(coarse_mse)}
        for mask in ['far_mask','halo_mask']:
            row[mask.replace('_mask','_pressure_mse')] = (float(error[g[mask],2].square().mean())
                                                        if g[mask].any() else None)
        rows.append(row)
    summary = {}
    for key in rows[0]:
        if key != 'seed':
            values = [r[key] for r in rows if r[key] is not None]
            summary[key] = np.mean(values,axis=0).tolist() if values else None
    summary['lattice_rmse'] = (np.sqrt(summary['mse'])*SCALE).tolist()
    summary['case_count'] = len(rows)
    return {'summary': summary, 'cases': rows}

def save_checkpoint(path, model, optimizer, scheduler, epoch, loss_fn, rng, details):
    memory = {'architecture': 'MultiscaleGNN', 'model_config': model.config,
              'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
              'scheduler_state': scheduler.state_dict(), 'completed_epochs': epoch,
              'loss_config': {'field_variance': loss_fn.field_variance.cpu().tolist(),
                              'edge_energy': loss_fn.edge_energy.cpu().tolist(),
                              'gradient_weight': loss_fn.gradient_weight,
                              'coarse_weight': loss_fn.coarse_weight},
              'loader_rng_state': rng.get_state(), 'torch_rng_state': torch.get_rng_state(),
              'cuda_rng_states': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
              'features': list(FEATURES), 'target_scale': SCALE.tolist(), 'target_offset': OFFSET.tolist(),
              'details': details}
    temp = path.with_suffix('.tmp')
    torch.save(memory, temp)
    temp.replace(path)

def load_new(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MultiscaleGNN(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state'])
    return model, FlowLoss(**checkpoint['loss_config']).to(device), checkpoint

def make_plots(model, graphs, out, baseline_model=None, count=5):
    from evaluate_dataset import reconstruct_grid, plot_fields
    out.mkdir(parents=True, exist_ok=True)
    indices = np.random.default_rng(42).choice(len(graphs), min(count,len(graphs)), replace=False)
    model.eval()
    for i in indices:
        g = graphs[i]
        with torch.inference_mode():
            values = {'solver': g['y'].cpu().numpy()*SCALE+OFFSET,
                      'multiscale': model(g).cpu().numpy()*SCALE+OFFSET}
            if baseline_model is not None:
                values['old_gnn'] = predict(baseline_model,g,True).cpu().numpy()*SCALE+OFFSET
        limits = (min(v[:,2].min() for v in values.values()), max(v[:,2].max() for v in values.values()))
        speed = max(np.percentile(np.linalg.norm(v[:,:2],axis=1),95) for v in values.values())
        solid = g['solid'].cpu().numpy()
        positions = g['levels'][0]['positions'].cpu().numpy()
        for label,v in values.items():
            grid = reconstruct_grid(v,positions,solid)
            plot_fields(grid,solid,out/f'seed_{g["seed"]}_{label}.png',
                        title=f'Seed {g["seed"]} | {label}', pressure_limits=limits,
                        arrow_scale=speed/2.4,reference_velocity=.01,
                        footer='Shared pressure colors and velocity arrow scale for this geometry.')
        error = values['multiscale']-values['solver']
        limit = max(abs(error[:,2]).max(),1e-12)
        grid = reconstruct_grid(error,positions,solid)
        plot_fields(grid,solid,out/f'seed_{g["seed"]}_error.png',
                    title=f'Seed {g["seed"]} | multiscale minus solver', pressure_limits=(-limit,limit),
                    arrow_scale=max(speed/2.4,1e-12),reference_velocity=.01,residual=True,
                    footer='Signed pressure error; arrows show velocity error.')

def main(args):
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = ROOT/'training_runs'/args.tag
    if out.exists() and not args.resume and not args.final:
        raise FileExistsError(f'Refusing to overwrite experiment {out}')
    out.mkdir(parents=True,exist_ok=True)
    train, train_inventory = get_graphs('train',args.levels,device)
    validation, validation_inventory = get_graphs('validation',args.levels,device)
    assert not {g['seed'] for g in train} & {g['seed'] for g in validation}
    print(f'Cached {len(train)} training and {len(validation)} validation graphs on {device}; '+
          f'example hierarchy {[len(l["weights"]) for l in train[0]["levels"]]}',flush=True)
    variance, energy = fit_loss_scales(train)
    loss_fn = FlowLoss(variance, energy, args.gradient_weight,args.coarse_weight).to(device)
    old_checkpoint = torch.load(BASELINE,map_location=device,weights_only=True)
    old_model = GNN(old_checkpoint['hidden'],old_checkpoint['layers'],old_checkpoint['global_context']).to(device)
    old_model.load_state_dict(old_checkpoint['model_state'])
    old_model.eval()
    if args.final:
        model, loss_fn, checkpoint = load_new(out/'best.pt',device)
        test, test_inventory = get_graphs('test',args.levels,device)
        assert not {g['seed'] for g in test} & {g['seed'] for g in train+validation}
        report = {'selected_epoch': checkpoint['completed_epochs'], 'test_inventory': test_inventory,
                  'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),
                  'baseline_sha256':hashlib.sha256(BASELINE.read_bytes()).hexdigest()}
        for split,graphs in [('train',train),('validation',validation),('test',test)]:
            report[split] = {'multiscale':evaluate(model,graphs,loss_fn),
                             'old_gnn':evaluate(old_model,graphs,loss_fn,True)}
            print(split, json.dumps({k:v['summary'] for k,v in report[split].items()}),flush=True)
        write_json(out/'final_metrics.json',report)
        make_plots(model,test,out/'test_plots',old_model)
        # Revisit the original reported pressure halo on known training geometry.
        make_plots(model,[g for g in train if g['seed']==1229],out/'original_halo_case',old_model,count=1)
        return
    baseline = evaluate(old_model,validation,loss_fn,True)
    write_json(out/'baseline_validation.json',baseline)
    print('Old GNN validation:',json.dumps(baseline['summary']),flush=True)
    model = MultiscaleGNN(hidden=args.hidden,levels=args.levels,local_steps=args.local_steps,
                         coarse_steps=args.coarse_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=args.lr*.05)
    rng = torch.Generator().manual_seed(args.seed)
    history, start, best = [], 0, float('inf')
    details = {'train':train_inventory,'validation':validation_inventory,'args':vars(args),
               'baseline_checkpoint':str(BASELINE), 'baseline_sha256':hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
               'selection':'minimum validation field + gradient matching + coarse-pressure objective',
               'success_criteria': 'lower held-out field and pressure-gradient errors, lower far-field pressure error, and no fixed-radius halo in inspected fields',
               'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                   ['multiscale_graph.py','multiscale_model.py','train_multiscale.py']}}
    if args.resume:
        checkpoint = torch.load(out/'last.pt',map_location=device,weights_only=True)
        if checkpoint['details']['train']['fingerprint'] != train_inventory['fingerprint']:
            raise ValueError('Training data changed since checkpoint')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        loss_fn = FlowLoss(**checkpoint['loss_config']).to(device)
        rng.set_state(checkpoint['loader_rng_state'].cpu())
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng_states']])
        start = checkpoint['completed_epochs']
        history = json.loads((out/'history.json').read_text())
        best = min(row['validation']['objective'] for row in history if 'validation' in row)
    if args.smoke:
        train = train[:4]
        initial = evaluate(model,train,loss_fn)['summary']
        write_json(out/'smoke_initial.json',initial)
    write_json(out/'config.json',details)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; loss variance {variance.tolist()}; '+
          f'edge scales {energy.tolist()}',flush=True)
    for epoch in range(start+1,args.epochs+1):
        begin = time.perf_counter()
        model.train()
        indices = torch.randperm(len(train),generator=rng).tolist()
        accumulated = np.zeros(4)
        for i in range(0,len(indices),args.batch_size):
            batch = indices[i:i+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for j in batch:
                graph = train[j]
                prediction = model(graph)
                loss, parts = loss_fn(prediction,graph)
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss epoch {epoch}, seed {graph["seed"]}')
                (loss/len(batch)).backward()
                accumulated += np.r_[float(loss.detach()),parts.cpu().numpy()]
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        row = {'epoch':epoch,'training_objective':float(accumulated[0]/len(train)),
               'training_parts':(accumulated[1:]/len(train)).tolist(),
               'lr':scheduler.get_last_lr()[0]}
        if epoch == 1 or epoch % args.evaluate_every == 0 or epoch == args.epochs:
            evaluation = evaluate(model,train if args.smoke else validation,loss_fn)
            row['validation' if not args.smoke else 'smoke_training'] = evaluation['summary']
            score = evaluation['summary']['objective']
            if score < best:
                best = score
                save_checkpoint(out/'best.pt',model,optimizer,scheduler,epoch,loss_fn,rng,details)
                write_json(out/'best_metrics.json',evaluation)
            save_checkpoint(out/'last.pt',model,optimizer,scheduler,epoch,loss_fn,rng,details)
            print(f'Epoch {epoch}: objective {row["training_objective"]:.6f}; '+
                  f'{"fit" if args.smoke else "validation"} MSE {evaluation["summary"]["loss"]:.7f}; '+
                  f'pressure {evaluation["summary"]["mse"][2]:.7f}; grad-p {evaluation["summary"]["gradient_mse"][2]:.7f}; '+
                  f'score {score:.5f}',flush=True)
        else:
            print(f'Epoch {epoch}: objective {row["training_objective"]:.6f}',flush=True)
        row['seconds'] = time.perf_counter()-begin
        history.append(row)
        write_json(out/'history.json',history)
    if args.smoke:
        final = evaluate(model,train,loss_fn)['summary']
        write_json(out/'smoke_result.json',{'initial':initial,'final':final,
                   'loss_ratio':final['loss']/initial['loss']})
        print('Smoke fit:',json.dumps(final),flush=True)
    else:
        best_model,_,_ = load_new(out/'best.pt',device)
        make_plots(best_model,validation,out/'validation_plots',old_model)
        print(f'Finished. Best checkpoint: {out/"best.pt"}',flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default='multiscale_v1')
    p.add_argument('--epochs',type=int,default=160)
    p.add_argument('--hidden',type=int,default=64)
    p.add_argument('--levels',type=int,default=5)
    p.add_argument('--local-steps',type=int,default=2)
    p.add_argument('--coarse-steps',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--gradient-weight',type=float,default=.15)
    p.add_argument('--coarse-weight',type=float,default=.25)
    p.add_argument('--seed',type=int,default=512)
    p.add_argument('--evaluate-every',type=int,default=5)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--final',action='store_true')
    main(p.parse_args())
