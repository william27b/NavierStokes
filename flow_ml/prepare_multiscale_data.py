"""Generate independent validation/test archives without changing training data."""
import contextlib
import hashlib
import json
from pathlib import Path
import numpy as np
from dataset import sample_case
from simulation import D2Q9Solver

ROOT = Path(__file__).resolve().parent

def prepare(count=40):
    training = sorted((ROOT / 'solver_runs').glob('*/metadata.json'))
    config = json.loads(training[0].read_text())['generation']['config']
    used = {json.loads(p.read_text())['generation']['seed'] for p in training}
    masks = set()
    for p in training:
        with np.load(p.parent/'fields.npz') as f:
            masks.add(hashlib.sha256(f['solid'].tobytes()).hexdigest())
    solver = D2Q9Solver()
    report = {}
    for split, first_seed in [('validation', 2000), ('test', 3000)]:
        dest = ROOT / 'heldout_runs' / split
        dest.mkdir(parents=True, exist_ok=True)
        records = []
        for path in sorted(dest.glob('*/metadata.json')):
            m = json.loads(path.read_text())
            records.append({'seed': m['generation']['seed'], 'run_id': path.parent.name,
                            **m['screening']})
            used.add(m['generation']['seed'])
            with np.load(path.parent/'fields.npz') as f:
                digest = hashlib.sha256(f['solid'].tobytes()).hexdigest()
            if digest in masks:
                raise ValueError('Duplicate geometry across split archives')
            masks.add(digest)
        seed = max([first_seed-1] + [r['seed'] for r in records]) + 1
        attempts = []
        with (dest/'generation.log').open('a', buffering=1) as log:
            while len(records) < count:
                if seed in used:
                    raise ValueError('Overlapping generation seeds')
                case = sample_case(np.random.default_rng(seed), config)
                digest = hashlib.sha256(case['solid'].tobytes()).hexdigest()
                if digest in masks:
                    seed += 1
                    continue
                with contextlib.redirect_stdout(log):
                    result = solver.solve(case['solid'], f'{split}_seed_{seed}',
                        **case['flow'], save=False, extra_metadata={'generation': {
                            'seed': seed, 'config': config, 'sampling': case['sampling']},
                            'split': split})
                fluid = ~result['solid']
                mach = float(np.linalg.norm(result['u'][:, fluid], axis=0).max()*np.sqrt(3))
                density = float(np.abs(result['rho'][fluid]-1).max())
                accepted = bool(result['converged'] and result['metadata']['invalid_fluid_cells'] == 0
                                and mach <= .1 and density <= .05)
                screen = {'max_mach': mach, 'max_density_deviation': density, 'accepted': accepted}
                attempts.append({'seed': seed, **screen})
                if accepted:
                    out = dest / result['metadata']['run_id']
                    out.mkdir(exist_ok=False)
                    metadata = result.pop('metadata')
                    metadata['screening'] = screen
                    np.savez_compressed(out/'fields.npz', **result)
                    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
                    records.append({'seed': seed, 'run_id': out.name, **screen})
                    masks.add(digest)
                    used.add(seed)
                    print(f'{split}: {len(records)}/{count}; seed {seed}; Mach {mach:.4f}', flush=True)
                else:
                    print(f'Rejected {split} seed {seed}: {screen}', flush=True)
                seed += 1
                (dest/'manifest.json').write_text(json.dumps({'split': split, 'config': config,
                    'records': records, 'attempts_this_run': attempts, 'next_seed': seed}, indent=2))
        report[split] = {'count': len(records), 'seeds': [r['seed'] for r in records]}
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    prepare()
