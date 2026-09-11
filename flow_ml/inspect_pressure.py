"""Pressure-only comparison and channel-pressure profile for a specified case."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiscale_graph import load_graph,to_device,SCALE,OFFSET
from train_multiscale import ROOT,BASELINE,load_new,predict
from model import GNN

def inspect(experiment,seed=1229,split='train'):
    folder=ROOT/'solver_runs' if split=='train' else ROOT/'heldout_runs'/split
    paths=[p.parent for p in folder.glob('*/metadata.json')
           if json.loads(p.read_text())['generation']['seed']==seed]
    if len(paths)!=1:
        raise ValueError('Expected exactly one matching geometry')
    model,_,checkpoint=load_new(experiment/'best.pt','cuda')
    model.eval()
    g=to_device(load_graph(paths[0]),'cuda')
    memory=torch.load(BASELINE,map_location='cuda',weights_only=True)
    baseline=GNN(memory['hidden'],memory['layers'],memory['global_context']).cuda().eval()
    baseline.load_state_dict(memory['model_state'])
    with torch.inference_mode():
        values={'Solver':g['y'][:,2].cpu().numpy(),
                'Previous GNN':predict(baseline,g,True)[:,2].cpu().numpy(),
                'Multiscale GNN':model(g)[:,2].cpu().numpy()}
    solid=g['solid'].cpu().numpy()
    fields={}
    for name,p in values.items():
        grid=np.full(solid.shape,np.nan)
        grid[~solid]=p*SCALE[2]
        fields[name]=grid
    # This supplementary view deliberately resolves interior variation. The
    # full-range pressure/quiver plots remain available alongside it.
    limits=np.percentile(fields['Solver'][~solid],[1,99])
    out=experiment/'pressure_detail'/f'{split}_seed_{seed}'
    out.mkdir(parents=True,exist_ok=True)
    for name,grid in fields.items():
        fig,ax=plt.subplots(figsize=(9,7.5),layout='constrained')
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#20252b')
        im=ax.imshow(grid,origin='lower',cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
        fig.colorbar(im,ax=ax,extend='both',label='Pressure minus outlet pressure (lattice)')
        ax.set(title=f'{name} | seed {seed}',xlabel='x (cells)',ylabel='y (cells)')
        fig.supxlabel('Shared solver 1st–99th percentile color limits; values outside are saturated.\nFull-range plots are provided separately.',fontsize=10)
        fig.savefig(out/(name.lower().replace(' ','_')+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for (name,grid),color in zip(fields.items(),['#111827','#d97706','#2563eb']):
        ax.plot(np.nanmean(grid[1:-1],axis=0),label=name,color=color,linewidth=2)
    ax.set(xlabel='x (cells)',ylabel='Mean fluid pressure minus outlet pressure',
           title=f'Pressure drop along the channel | seed {seed}')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'channel_pressure_profile.png',dpi=160);plt.close(fig)
    (out/'details.json').write_text(json.dumps({'seed':seed,'split':split,
        'checkpoint_epoch':checkpoint['completed_epochs'],'color_limits':limits.tolist()},indent=2))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('experiment',type=Path)
    p.add_argument('--seed',type=int,default=1229)
    p.add_argument('--split',default='train',choices=['train','validation','test'])
    args=p.parse_args();inspect(args.experiment,args.seed,args.split)
