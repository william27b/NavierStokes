"""Boundary-conditioned graph U-Net with fluid-connected restriction/prolongation."""
import torch
from torch import nn
from multiscale_graph import FEATURES, pool

def mlp(inputs, outputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

class Processor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.message = mlp(2*hidden+2, hidden, hidden)
        self.update = mlp(2*hidden, hidden, hidden)

    def forward(self, h, level):
        z = self.norm(h)
        src, dst = level['edges']
        message = self.message(torch.cat((z[src], z[dst], level['edge_attr']), dim=-1))
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, dst, message)
        degree = torch.bincount(dst, minlength=len(h)).clamp_min(1).to(h.dtype)
        aggregate = aggregate / degree[:, None]
        return h + self.update(torch.cat((z, aggregate), dim=-1))

class MultiscaleGNN(nn.Module):
    def __init__(self, hidden=64, levels=5, local_steps=2, coarse_steps=8, hard_boundaries=True):
        super().__init__()
        self.config = dict(hidden=hidden, levels=levels, local_steps=local_steps,
                           coarse_steps=coarse_steps, hard_boundaries=hard_boundaries)
        self.encoder = mlp(len(FEATURES), hidden, hidden)
        self.down = nn.ModuleList([nn.ModuleList([Processor(hidden) for _ in range(local_steps)])
                                   for _ in range(levels-1)])
        self.restrict = nn.ModuleList([mlp(hidden+len(FEATURES)+1, hidden, hidden) for _ in range(levels-1)])
        self.coarse = nn.ModuleList([Processor(hidden) for _ in range(coarse_steps)])
        self.prolong = nn.ModuleList([mlp(2*hidden, hidden, hidden) for _ in range(levels-1)])
        self.up = nn.ModuleList([nn.ModuleList([Processor(hidden) for _ in range(local_steps)])
                                 for _ in range(levels-1)])
        self.decoder = mlp(hidden, 3, hidden)

    def forward(self, graph):
        levels = graph['levels']
        if len(levels) != self.config['levels']:
            raise ValueError('Graph and model hierarchy depths differ')
        h = self.encoder(graph['x'])
        skip = []
        for k, processors in enumerate(self.down):
            for processor in processors:
                h = processor(h, levels[k])
            skip.append(h)
            coarse = pool(h, levels[k], levels[k+1])
            h = coarse + self.restrict[k](torch.cat((coarse, levels[k+1]['geom']), dim=-1))
        for processor in self.coarse:
            h = processor(h, levels[-1])
        for k in range(len(skip)-1, -1, -1):
            lifted = h[levels[k]['mapping']]
            h = skip[k] + lifted + self.prolong[k](torch.cat((skip[k], lifted), dim=-1))
            for processor in self.up[k]:
                h = processor(h, levels[k])
        prediction = self.decoder(h)
        if self.config['hard_boundaries']:
            prediction = torch.where(graph['bc_mask'], graph['bc_values'], prediction)
        return prediction

class FlowLoss(nn.Module):
    """Field fit, derivative matching, and coarse pressure fit; training-only scales."""
    def __init__(self, field_variance, edge_energy, gradient_weight=.15, coarse_weight=.25):
        super().__init__()
        self.register_buffer('field_variance', torch.as_tensor(field_variance).float().clamp_min(1e-8))
        self.register_buffer('edge_energy', torch.as_tensor(edge_energy).float().clamp_min(1e-8))
        self.gradient_weight = gradient_weight
        self.coarse_weight = coarse_weight

    def forward(self, prediction, graph):
        error = prediction - graph['y']
        field = (error.square().mean(0) / self.field_variance).mean()
        a, b = graph['levels'][0]['edges']
        gradient = ((error[b]-error[a]).square().mean(0) / self.edge_energy).mean()
        coarse = []
        pressure = error[:, 2:3]
        for fine, coarser in zip(graph['levels'][:-1], graph['levels'][1:]):
            pressure = pool(pressure, fine, coarser)
            coarse.append((pressure[:, 0].square()*coarser['weights']).sum()/coarser['weights'].sum())
        regional = torch.stack(coarse).mean()/self.field_variance[2] if coarse else field*0
        total = field + self.gradient_weight*gradient + self.coarse_weight*regional
        return total, torch.stack((field, gradient, regional)).detach()

def fit_loss_scales(graphs):
    mean = torch.stack([g['y'].mean(0) for g in graphs]).mean(0)
    variance = torch.stack([(g['y']-mean).square().mean(0) for g in graphs]).mean(0)
    edge_energy = []
    for graph in graphs:
        a,b = graph['levels'][0]['edges']
        edge_energy.append((graph['y'][b]-graph['y'][a]).square().mean(0))
    return variance.cpu(), torch.stack(edge_energy).mean(0).cpu()
