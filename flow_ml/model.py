from torch.nn import Sequential, Linear, SiLU
from torch import nn
import torch

def make_mlp(in_dim, out_dim, hidden=64):
    return Sequential(
        Linear(in_dim, hidden),
        SiLU(),
        Linear(hidden, out_dim)
    )

class MessagePassing(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.message = make_mlp(2 * hidden + 2, hidden)
        self.update = make_mlp(2 * hidden, hidden)

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        aggregate = torch.concat((h[src], h[dst], edge_attr), dim=-1)
        messages = self.message(aggregate)

        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)

        update = self.update(torch.concat((h, aggregated), dim=-1))
        return h + update

class GNN(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()

        self.encoder = make_mlp(2, hidden, hidden)
        self.processors = nn.ModuleList([MessagePassing(hidden) for _ in range(layers)])
        self.decoder = make_mlp(hidden, 3, hidden)

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)

        for processor in self.processors:
            h = processor(h, edge_index, edge_attr)

        return self.decoder(h)